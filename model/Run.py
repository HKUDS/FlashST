
import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
from datetime import datetime
from Trainer import Trainer
from FlashST import FlashST as Network_Pretrain
from FlashST import FlashST as Network_Predict
from lib.TrainInits import init_seed
from lib.TrainInits import print_model_parameters
from lib.metrics import MAE_torch, MSE_torch, huber_loss
from lib.predifineGraph import *
from lib.data_process import define_dataloder, get_val_tst_dataloader, data_type_init
from conf.FlashST.Params_pretrain import parse_args
import torch.nn.functional as F

# *************************************************************************#
# mode = 'eval'   # pretrain eval ori test
# dataset_test = ['CA_District5'] # NYC_BIKE, CA_District5, PEMS07M, chengdu_didi
# dataset_use = ['PEMS08', 'PEMS04', 'PEMS07', 'PEMS03'] # PEMS08, PEMS04, PEMS07, PEMS03
# model = 'STGCN'    # TGCN STGCN ASTGCN GWN STSGCN AGCRN MTGNN STFGNN STGODE DMSTGCN MSDR STWA PDFormer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = parse_args(device)

print('Mode: ', args.mode, '  model: ', args.model, '  DATASET: ', args.dataset_test,
      '  load_pretrain_path: ', args.load_pretrain_path, '  save_pretrain_path: ', args.save_pretrain_path)


def Mkdir(path):
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)

def infoNCEloss():
    def loss(q, k):
        T = 0.3
        pos_sim = torch.sum(torch.mul(q, q), dim=-1)
        neg_sim = torch.matmul(q, q.transpose(-1, -2))
        pos = torch.exp(torch.div(pos_sim, T))
        neg = torch.sum(torch.exp(torch.div(neg_sim, T)), dim=-1)
        denominator = neg + pos
        return torch.mean(-torch.log(torch.div(pos, denominator)))
    return loss

def scaler_mae_loss(mask_value):
    def loss(preds, labels, scaler, mask=None):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae, mae_loss = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        # print(mae.shape, mae_loss.shape)
        return mae, mae_loss
    return loss

def scaler_huber_loss(mask_value):
    def loss(preds, labels, scaler, mask=None):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae, mae_loss = huber_loss(pred=preds, true=labels, mask_value=mask_value)
        # print(mae.shape, mae_loss.shape)
        return mae, mae_loss
    return loss

if args.model == 'GWN' or args.model == 'MTGNN' or args.model == 'STFGNN' or args.model == 'STGODE' or args.model == 'DMSTGCN':
    seed_mode = False   # for quick running
else:
    seed_mode = True
init_seed(args.seed, seed_mode)

#config log path
current_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(current_dir, '../SAVE', args.mode, args.model)
Mkdir(log_dir)
args.log_dir = log_dir

#predefine Graph
dataset_graph = []
if args.mode == 'pretrain':
    dataset_graph = args.dataset_use.copy()
else:
    dataset_graph.append(args.dataset_test)
args.dataset_graph = dataset_graph
pre_graph_dict(args)
data_type_init(args.dataset_test, args)

if args.model == 'STGODE' or args.model == 'AGCRN' or args.model == 'ASTGCN':
    xavier = True
else:
    xavier = False

args.xavier = xavier

#load dataset
if args.mode == 'pretrain':
    x_trn_dict, y_trn_dict, _, _, _, _, scaler_dict = define_dataloder(stage='Train', args=args)
    eval_train_loader, eval_val_loader, eval_test_loader, eval_scaler_dict = None, None, None, None
else:
    x_trn_dict, y_trn_dict, scaler_dict = None, None, None
    eval_x_trn_dict, eval_y_trn_dict, eval_x_val_dict, eval_y_val_dict, eval_x_tst_dict, eval_y_tst_dict, eval_scaler_dict = define_dataloder(stage='eval', args=args)
    eval_train_loader = get_val_tst_dataloader(eval_x_trn_dict, eval_y_trn_dict, args, shuffle=True)
    eval_val_loader = get_val_tst_dataloader(eval_x_val_dict, eval_y_val_dict, args, shuffle=False)
    eval_test_loader = get_val_tst_dataloader(eval_x_tst_dict, eval_y_tst_dict, args, shuffle=False)


#init model
if args.mode == 'pretrain':
    model = Network_Pretrain(args)
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    model = model.to(args.device)
else:
    model = Network_Predict(args)
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    model = model.to(args.device)
    if args.mode == 'eval':
        load_dir = os.path.join(current_dir, '../SAVE', 'pretrain', args.model)
        model.load_state_dict(torch.load(load_dir + '/' + args.load_pretrain_path), strict=False)
        print(load_dir + '/' + args.load_pretrain_path)
        print('load pretrain model!!!')

print_model_parameters(model, only_num=False)

#init loss function, optimizer
if args.loss_func == 'mask_mae':
    if (args.model == 'STSGCN' or args.model == 'STFGNN' or args.model == 'STGODE'):
        loss = scaler_huber_loss(mask_value=args.mape_thresh)
        print('============================scaler_huber_loss')
    else:
        loss = scaler_mae_loss(mask_value=args.mape_thresh)
        print('============================scaler_mae_loss')
    # print(args.model, Mode)
elif args.loss_func == 'mae':
    loss = torch.nn.L1Loss().to(args.device)
elif args.loss_func == 'mse':
    loss = torch.nn.MSELoss().to(args.device)
else:
    raise ValueError


optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                             weight_decay=0, amsgrad=False)
#learning rate decay
lr_scheduler = None
if args.lr_decay:
    print('Applying learning rate decay.')
    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=lr_decay_steps,
                                                        gamma=args.lr_decay_rate)

#start training
loss_mse = torch.nn.MSELoss().to(args.device)
loss_ssl = infoNCEloss()
trainer = Trainer(model, loss, loss_ssl, optimizer, x_trn_dict, y_trn_dict, args.A_dict, args.lpls_dict, eval_train_loader,
                       eval_val_loader, eval_test_loader, scaler_dict, eval_scaler_dict, args,
                       lr_scheduler=lr_scheduler)

if args.mode == 'pretrain':
    trainer.train_pretrain()
elif args.mode == 'eval':
    trainer.train_eval()
elif args.mode == 'ori':
    trainer.train_eval()
elif args.mode == 'test':
    # model.load_state_dict(torch.load(log_dir + '/' + args.load_pretrain_path), strict=True)
    # print("Load saved model")
    trainer.eval_test(model, trainer.args, args.A_dict, args.lpls_dict, eval_test_loader, eval_scaler_dict[args.dataset_test], trainer.logger)
else:
    raise ValueError
