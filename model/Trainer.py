import torch
import math
import os
import time
import copy
import numpy as np
from lib.logger import get_logger
from lib.metrics import All_Metrics
from tqdm import tqdm
from lib.data_process import get_pretrain_task_batch


class Trainer(object):
    def __init__(self, model, loss, loss_ssl, optimizer, x_trn_dict, y_trn_dict, A_dict, lpls_dict, eval_train_dataloader,
                       eval_val_dataloader, eval_test_dataloader, scaler_dict, eval_scaler_dict,
                  args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.args = args
        self.loss = loss
        self.loss_ssl = loss_ssl
        self.optimizer = optimizer
        self.x_trn_dict, self.y_trn_dict = x_trn_dict, y_trn_dict
        self.A_dict, self.lpls_dict = A_dict, lpls_dict
        self.eval_train_loader = eval_train_dataloader
        self.eval_val_loader = eval_val_dataloader
        self.eval_test_loader = eval_test_dataloader
        self.scaler_dict = scaler_dict
        self.eval_scaler_dict = eval_scaler_dict
        self.lr_scheduler = lr_scheduler
        if eval_train_dataloader is not None:
            self.eval_train_per_epoch = len(eval_train_dataloader)
            self.eval_val_per_epoch = len(eval_val_dataloader)
        self.batch_seen = 0
        if eval_val_dataloader != None:
            self.val_per_epoch = len(eval_val_dataloader)
        self.best_path = os.path.join(self.args.log_dir, self.args.model + '_' + self.args.dataset_test + '.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        #log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        #if not args.debug:
        #self.logger.info("Argument: %r", args)
        # for arg, value in sorted(vars(args).items()):
        #     self.logger.info("Argument %s: %r", arg, value)

    # def save_checkpoint(self):
    #     state = {
    #         'state_dict': self.model.state_dict(),
    #         'optimizer': self.optimizer.state_dict(),
    #         'config': self.args
    #     }
    #     torch.save(state, self.best_path)
    #     self.logger.info("Saving current best model to " + self.best_path)


    def train_pretrain(self, ):
        """
        pretraining stage
        """
        for epoch in tqdm(range(self.args.pretrain_epochs)):
            start_time = time.time()
            spt_task_x, spt_task_y, select_dataset, train_len = get_pretrain_task_batch(self.args, self.x_trn_dict, self.y_trn_dict)
            print(select_dataset)
            loss, loss_pred, loss_ssl = self.train_pretrain_eps(spt_task_x, spt_task_y, select_dataset, train_len, epoch)
            end_time = time.time()
            if epoch % 1 == 0:
                print(
                    "[Source Train] epoch #{}/{}: loss is {} pred loss {} ssl loss {}, training time is {}".format(
                    epoch + 1, self.args.pretrain_epochs, round(loss, 2), round(loss_pred, 2), round(loss_ssl, 2),
                    round(end_time - start_time, 2)))
        print("Pre-train finish.")

    def train_pretrain_eps(self, spt_task_x, spt_task_y, select_dataset, train_len, epoch):
        self.model.train()
        total_loss = 0
        pred_loss = 0
        ssl_loss = 0

        nadj = self.A_dict[select_dataset]
        lpls = self.lpls_dict[select_dataset]

        for i in range(train_len):
            x_in, y_in, y_lbl = spt_task_x[i], spt_task_y[i], spt_task_y[i][..., 0:1]
            x_in, y_in, y_lbl = x_in.to(self.args.device), y_in.to(self.args.device), y_lbl.to(self.args.device)
            out, q = self.model(x_in, x_in, select_dataset, batch_seen=None, nadj=nadj, lpls=lpls, useGNN=True, DSU=True)
            loss_pred, _ = self.loss(out, y_lbl, self.scaler_dict[select_dataset])
            loss_ssl = self.loss_ssl(q, q)
            loss = loss_pred + loss_ssl
            # loss = loss_pred
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            pred_loss += loss_pred.item()
            ssl_loss += loss_ssl.item()

        train_epoch_loss = total_loss / train_len
        train_pred_loss = pred_loss / train_len
        train_ssl_loss = ssl_loss / train_len

        if train_pred_loss + train_ssl_loss < 30:
            best_premodel = copy.deepcopy(self.model.state_dict())
            torch.save(best_premodel, self.best_path+'_ep'+str(epoch)+'_EL'+str(round(train_epoch_loss, 2))+'_PL'+str(round(train_pred_loss, 2))+'.pth')

        self.logger.info("Saving current best model to " + self.best_path)
        return train_epoch_loss, train_pred_loss, train_ssl_loss

    def train_eval(self):
        """
        prompt-tunning stage
        """
        # best_model_test = None
        # train_loss_list = []
        # val_loss_list = []

        best_loss = float('inf')
        not_improved_count = 0
        if self.args.mode == 'eval':
            eps = self.args.eval_epochs
        else:
            eps = self.args.ori_epochs
        # for epoch in range(eps):
        for epoch in tqdm(range(eps)):
            start_time = time.time()
            train_epoch_loss, loss_pre = self.eval_trn_eps()
            end_time = time.time()
            print('time cost: ', round(end_time-start_time, 2))
            val_epoch_loss = self.eval_val_eps()
            print("[Target Fine-tune] epoch #{}/{}: loss is {}, val_loss is {}".format(
                epoch+1, eps, round(train_epoch_loss, 2), round(val_epoch_loss, 2)))
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False

            # train_loss_list.append(loss_pre.detach().cpu().numpy())
            # val_loss_list.append(val_epoch_loss)

            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break

            # save the best state
            if best_state == True:
                best_model = copy.deepcopy(self.model.state_dict())
                # best_model_test = copy.deepcopy(self.model)
                torch.save(best_model, self.best_path)
                self.logger.info('*********************************Current best model saved!')

        # #save the best model to file
        # if self.args.debug:
        #     torch.save(best_model, self.best_path)
        #     self.logger.info("Saving current best model to " + self.best_path)
        # test
        self.model.load_state_dict(best_model)
        #self.val_epoch(self.args.epochs, self.test_loader)
        self.eval_test(self.model, self.args, self.A_dict, self.lpls_dict, self.eval_test_loader, self.eval_scaler_dict[self.args.dataset_test], self.logger)


    def eval_trn_eps(self):
        self.model.train()
        total_loss = 0
        nadj = self.A_dict[self.args.dataset_test]
        lpls = self.lpls_dict[self.args.dataset_test]
        for batch_idx, (data, target) in enumerate(self.eval_train_loader):
            self.batch_seen += 1
            data = data.to(self.args.device)
            target = target.to(self.args.device)
            data = data[..., :self.args.input_base_dim + self.args.input_extra_dim]
            label = target[..., :self.args.input_base_dim + self.args.input_extra_dim]
            out, q = self.model(data, data, self.args.dataset_test, self.batch_seen, nadj=nadj, lpls=lpls, useGNN=True, DSU=True)
            loss_pred, _ = self.loss(out, label[..., :self.args.output_dim], self.eval_scaler_dict[self.args.dataset_test])

            if self.args.mode == 'eval':
                loss_ssl = self.loss_ssl(q, q)
                loss = loss_ssl + loss_pred
            else:
                loss = loss_pred
            self.optimizer.zero_grad()
            loss.backward()
            # add max grad clipping
            # if self.args.grad_norm:
            #     torch.nn.utils.clip_grad_norm_(maml_model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
        train_epoch_loss = total_loss / self.eval_train_per_epoch

        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss, loss_pred


    def eval_val_eps(self):
        self.model.eval()
        total_val_loss = 0
        nadj = self.A_dict[self.args.dataset_test]
        lpls = self.lpls_dict[self.args.dataset_test]
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.eval_val_loader):
                data = data.to(self.args.device)
                target = target.to(self.args.device)
                data = data[..., :self.args.input_base_dim + self.args.input_extra_dim]
                label = target[..., :self.args.input_base_dim + self.args.input_extra_dim]
                out, _ = self.model(data, data, self.args.dataset_test, batch_seen=None, nadj=nadj, lpls=lpls, useGNN=True, DSU=False)
                loss, _ = self.loss(out, label[..., :self.args.output_dim], self.eval_scaler_dict[self.args.dataset_test])
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_epoch_loss = total_val_loss / self.eval_val_per_epoch
        # self.eval_test(self.model, self.args, self.A_dict, self.lpls_dict, self.eval_test_loader,
        #                self.eval_scaler_dict[self.args.dataset_test], self.logger)
        return val_epoch_loss

    @staticmethod
    def eval_test(model, args, A_dict, lpls_dict, data_loader, scaler, logger, path=None):
        nadj = A_dict[args.dataset_test]
        lpls = lpls_dict[args.dataset_test]
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data.to(args.device)
                target = target.to(args.device)
                data = data[..., :args.input_base_dim + args.input_extra_dim]
                output, _ = model(data, data, args.dataset_test, batch_seen=None, nadj=nadj, lpls=lpls, useGNN=True, DSU=False)
                label = target[..., :args.output_dim]
                y_true.append(label)
                y_pred.append(output)
        if not args.real_value:
            y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
        # np.save('./{}_true.npy'.format(args.dataset+'_'+args.model+'_'+args.mode), y_true.cpu().numpy())
        # np.save('./{}_pred.npy'.format(args.dataset+'_'+args.model+'_'+args.mode), y_pred.cpu().numpy())
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                t + 1, mae, rmse, mape*100))
        mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                    mae, rmse, mape*100))
        return mae
