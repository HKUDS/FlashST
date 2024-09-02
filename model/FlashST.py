import torch
import torch.nn as nn
import torch.nn.functional as F

class FlashST(nn.Module):
    def __init__(self, args):
        super(FlashST, self).__init__()
        self.num_node = args.num_nodes
        self.input_base_dim = args.input_base_dim
        self.input_extra_dim = args.input_extra_dim
        self.output_dim = args.output_dim
        self.his = args.his
        self.pred = args.pred
        self.embed_dim = args.embed_dim
        self.mode = args.mode
        self.model = args.model
        self.load_pretrain_path = args.load_pretrain_path
        self.log_dir = args.log_dir
        self.args = args

        if self.mode == 'ori':
            dim_in = self.input_base_dim
        else:
            dim_in = self.embed_dim*4
        args.dim_in = dim_in
        dim_out = self.output_dim

        if self.model == 'AGCRN':
            from model.AGCRN.AGCRN import AGCRN
            from model.AGCRN.args import parse_args
            args_predictor = parse_args(args.dataset_test)
            self.predictor = AGCRN(args_predictor, dim_in, dim_out, args.A_dict, args.dataset_use, args.dataset_test, self.mode)
        elif self.model == 'MTGNN':
            from model.MTGNN.MTGNN import MTGNN
            from model.MTGNN.args import parse_args
            args_predictor = parse_args(args.dataset_test)
            self.predictor = MTGNN(args_predictor, dim_in, dim_out, args.A_dict, args.dataset_use, args.dataset_test, self.mode)
        elif self.model == 'STGCN':
            from model.STGCN.stgcn import STGCN
            from model.STGCN.args import parse_args
            args_predictor = parse_args(args.dataset_test, args)
            self.predictor = STGCN(args_predictor, args_predictor.G_dict, args.dataset_use, [args.dataset_test], args.num_nodes, dim_in, dim_out, args.device, self.mode)
        elif self.model == 'STSGCN':
            from model.STSGCN.STSGCN import STSGCN
            from model.STSGCN.args import parse_args
            args_predictor = parse_args(args.dataset_test)
            self.predictor = STSGCN(args_predictor, args.num_nodes, args.his, dim_in, dim_out, args.A_dict, args.dataset_use, args.dataset_test, args.device, self.mode)
        elif self.model == 'ASTGCN':
            from model.ASTGCN.ASTGCN import ASTGCN
            from model.ASTGCN.args import parse_args
            args_predictor = parse_args(args.dataset_test, args)
            self.predictor = ASTGCN(args_predictor, args.A_dict[args.dataset_test], args_predictor.num_nodes, args_predictor.len_input, args_predictor.num_for_predict, dim_in, dim_out, args.device)
        elif self.model == 'GWN':
            from model.GWN.GWN import GWNET
            # from GWN.GWNori import gwnet
            from model.GWN.args import parse_args
            args_predictor = parse_args(args.dataset_test)
            self.predictor = GWNET(args_predictor, dim_in, dim_out, args.A_dict, args.dataset_use, args.dataset_test, self.mode)
        elif self.model == 'DMSTGCN':
            from model.DMSTGCN.DMSTGCN import DMSTGCN
            self.predictor = DMSTGCN(args.device, dim_in, args.A_dict, args.dataset_use, args.dataset_test, self.mode)
        elif self.model == 'TGCN':
            from model.TGCN.TGCN import TGCN
            from model.TGCN.args import parse_args
            args_predictor = parse_args(args.dataset_test)
            self.predictor = TGCN(args_predictor, args.A_dict, args.dataset_test, args.device, dim_in)
        elif self.model == 'STFGNN':
            from model.STFGNN.STFGNN import STFGNN
            from model.STFGNN.args import parse_args
            args_predictor = parse_args(args.dataset_test, args)
            self.predictor = STFGNN(args_predictor, dim_in)
        elif self.model == 'STGODE':
            from model.STGODE.STGODE import ODEGCN
            from model.STGODE.args import parse_args
            args_predictor = parse_args(args.dataset_test, args)
            self.predictor = ODEGCN(args.num_nodes, dim_in, args.his, args.pred, args_predictor.A_sp_wave_dict,
                                    args_predictor.A_se_wave_dict, dim_out, args.A_dict, args.dataset_use, args.dataset_test, self.mode, args.device)
        elif self.model == 'STWA':
            from model.ST_WA.ST_WA import STWA
            from model.ST_WA.args import parse_args
            args_predictor = parse_args(args.dataset_test)
            self.predictor = STWA(args_predictor.device, args_predictor.num_nodes, dim_in, args_predictor.out_dim,
                                  args_predictor.channels, args_predictor.dynamic, args_predictor.lag,
                                  args_predictor.horizon, args_predictor.supports, args_predictor.memory_size, args.A_dict, args.dataset_use, args.dataset_test, self.mode)
        elif self.model == 'MSDR':
            from model.MSDR.gmsdr_model import GMSDRModel
            from model.MSDR.args import parse_args
            args_predictor = parse_args(args.dataset_test)
            args_predictor.input_dim = dim_in
            args_predictor.A_dict = args.A_dict
            args_predictor.dataset_use = args.dataset_use
            args_predictor.dataset_test = args.dataset_test
            args_predictor.mode = args.mode
            self.predictor = GMSDRModel(args_predictor)

        elif self.model == 'PDFormer':
            from model.PDFormer.PDFformer import PDFormer
            from model.PDFormer.args import parse_args
            args_predictor = parse_args(args.dataset_test, args)
            self.predictor = PDFormer(args_predictor, args)

        if self.mode == 'eval':
            for param in self.predictor.parameters():
                param.requires_grad = False
            # STGCN #
            if self.model == 'STGCN':
                for param in self.predictor.st_conv1.ln_eval.parameters():
                    param.requires_grad = True
                for param in self.predictor.st_conv2.ln_eval.parameters():
                    param.requires_grad = True
                for param in self.predictor.output.ln_eval.parameters():
                    param.requires_grad = True
                for param in self.predictor.output.fc_pretrain.parameters():
                    param.requires_grad = True
            # GWN #
            elif self.model == 'GWN':
                for param in self.predictor.nodevec1_eval:
                    param.requires_grad = True
                for param in self.predictor.nodevec2_eval:
                    param.requires_grad = True
                for param in self.predictor.end_conv_2.parameters():
                    param.requires_grad = True
            # MTGNN #
            elif self.model == 'MTGNN':
                for param in self.predictor.gc_eval.parameters():
                    param.requires_grad = True
                for param in self.predictor.norm_eval.parameters():
                    param.requires_grad = True
            # PDFormer #
            elif self.model == 'PDFormer':
                for param in self.predictor.end_conv1.parameters():
                    param.requires_grad = True
                for param in self.predictor.end_conv2.parameters():
                    param.requires_grad = True

        if (args.mode == 'pretrain' or args.mode == 'ori') and args.xavier:
            for p in self.predictor.parameters():
                if p.dim() > 1 and p.requires_grad:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.uniform_(p)

        if self.mode != 'ori':
            from PromptNet import PromptNet
            self.pretrain_model = PromptNet(args)


    def forward(self, source, label, select_dataset, batch_seen=None, nadj=None, lpls=None, useGNN=False):
        if self.mode == 'ori':
            return self.forward_ori(source, label, select_dataset, batch_seen)
        else:
            return self.forward_pretrain(source, label, select_dataset, batch_seen, nadj, lpls, useGNN)

    def forward_pretrain(self, source, label, select_dataset, batch_seen=None, nadj=None, lpls=None, useGNN=False):
        x_prompt_return = self.pretrain_model(source[..., :self.input_base_dim], source, None, nadj, lpls, useGNN)
        if self.model == 'DMSTGCN':
            x_predic = self.predictor(x_prompt_return, source[:, 0, 0, 1], select_dataset)  # MTGNN
        else:
            x_predic = self.predictor(x_prompt_return, select_dataset)   # STGCN
        return x_predic, x_prompt_return

    def forward_ori(self, source, label=None, select_dataset=None, batch_seen=None):
        if self.model == 'DMSTGCN':
            x_predic = self.predictor(source[..., :self.input_base_dim], source[:, 0, 0, 1], select_dataset)  # MTGNN
        else:
            x_predic = self.predictor(source[..., :self.input_base_dim], select_dataset)
        return x_predic, None
