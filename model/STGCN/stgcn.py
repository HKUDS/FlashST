import math
import numpy as np
from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)  # filter=(1,1)

    def forward(self, x):  # x: (batch_size, feature_dim(c_in), input_length, num_nodes)
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x  # return: (batch_size, c_out, input_length-1+1, num_nodes-1+1)

class TemporalConvLayer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(TemporalConvLayer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1, padding=[int((kt-1)/2), 0])
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1, padding=[int((kt-1)/2), 0])

    def forward(self, x):
        """

        :param x: (batch_size, feature_dim(c_in), input_length, num_nodes)
        :return: (batch_size, c_out, input_length-kt+1, num_nodes)
        """
        # x_in = self.align(x)[:, :, self.kt - 1:, :]  # (batch_size, c_out, input_length-kt+1, num_nodes)
        x_in = self.align(x)[:, :, :, :]  # (batch_size, c_out, input_length-kt+1, num_nodes)
        if self.act == "GLU":
            # x: (batch_size, c_in, input_length, num_nodes)
            x_conv = self.conv(x)
            # x_conv: (batch_size, c_out * 2, input_length-kt+1, num_nodes)  [P Q]
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
            # return P * sigmoid(Q) shape: (batch_size, c_out, input_length-kt+1, num_nodes)
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)  # residual connection
        return torch.relu(self.conv(x) + x_in)  # residual connection

class SpatioConvLayer(nn.Module):
    def __init__(self, ks, c_in, c_out, lk, device):
        super(SpatioConvLayer, self).__init__()
        self.Lk = lk
        self.device = device
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks).to(device))  # kernel: C_in*C_out*ks
        self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1, 1).to(device))
        self.align = Align(c_in, c_out)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x, select_dataset):
        # Lk: (Ks, num_nodes, num_nodes)
        # x:  (batch_size, c_in, input_length, num_nodes)
        # x_c: (batch_size, c_in, input_length, Ks, num_nodes)
        # theta: (c_in, c_out, Ks)
        # x_gc: (batch_size, c_out, input_length, num_nodes)
        # print(select_dataset, self.Lk[select_dataset])
        x_c = torch.einsum("knm,bitm->bitkn", self.Lk[select_dataset].to(self.device), x)  # delete num_nodes(n)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b  # delete Ks(k) c_in(i)
        x_in = self.align(x)  # (batch_size, c_out, input_length, num_nodes)
        return torch.relu(x_gc + x_in)  # residual connection


class STConvBlock(nn.Module):
    def __init__(self, ks, kt, n, c, p, lk, dataset_use, dataset_test, device, mode):
        super(STConvBlock, self).__init__()
        self.mode = mode
        self.tconv1 = TemporalConvLayer(kt, c[0], c[1], "GLU")
        self.sconv = SpatioConvLayer(ks, c[1], c[1], lk, device)
        self.tconv2 = TemporalConvLayer(kt, c[1], c[2])
        self.dataset2index = {}
        if self.mode == 'pretrain':
            self.ln_pretrain = nn.ModuleList()
            for i, data_graph in enumerate(dataset_use):
                self.dataset2index[data_graph] = i
                n_dataset = lk[data_graph].shape[1]
                self.ln_pretrain.append(nn.LayerNorm([n_dataset, c[2]]))
        else:
            self.ln_eval = nn.ModuleList()
            for i, data_graph in enumerate(dataset_test):
                self.dataset2index[data_graph] = i
                n_dataset = lk[data_graph].shape[1]
                self.ln_eval.append(nn.LayerNorm([n_dataset, c[2]]))
        self.dropout = nn.Dropout(p)

    def forward(self, x, select_dataset):  # x: (batch_size, feature_dim/c[0], input_length, num_nodes)
        x_t1 = self.tconv1(x)    # (batch_size, c[1], input_length-kt+1, num_nodes)
        x_s = self.sconv(x_t1, select_dataset)   # (batch_size, c[1], input_length-kt+1, num_nodes)
        x_t2 = self.tconv2(x_s)  # (batch_size, c[2], input_length-kt+1-kt+1, num_nodes)
        x_t2 = x_t2.permute(0, 2, 3, 1).permute(0, 3, 1, 2)
        if self.mode == 'pretrain':
            x_ln = self.ln_pretrain[self.dataset2index[select_dataset]](x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            x_ln = self.ln_eval[self.dataset2index[select_dataset]](x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x_ln)

class FullyConvLayer(nn.Module):
    def __init__(self, c, out_dim):
        super(FullyConvLayer, self).__init__()
        self.conv = nn.Conv2d(c, out_dim, 1)  # c,self.output_dim,1

    def forward(self, x):
        return self.conv(x)


class OutputLayer(nn.Module):
    def __init__(self, c, t, n, out_dim, lk, dataset_use, dataset_test, mode):
        super(OutputLayer, self).__init__()
        self.tconv1 = TemporalConvLayer(t, c, c, "GLU")
        self.dataset2index = {}
        self.mode = mode
        self.fc_pretrain = FullyConvLayer(c, out_dim)
        if self.mode == 'pretrain':
            self.ln_pretrain = nn.ModuleList()
            for i, data_graph in enumerate(dataset_use):
                self.dataset2index[data_graph] = i
                n_dataset = lk[data_graph].shape[1]
                self.ln_pretrain.append(nn.LayerNorm([n_dataset, c]))
        else:
            self.ln_eval = nn.ModuleList()
            for i, data_graph in enumerate(dataset_test):
                self.dataset2index[data_graph] = i
                n_dataset = lk[data_graph].shape[1]
                self.ln_eval.append(nn.LayerNorm([n_dataset, c]))
            # self.fc_eval = FullyConvLayer(c, out_dim)
        self.tconv2 = TemporalConvLayer(1, c, c, "sigmoid")  # kernel=1*1

    def forward(self, x, select_dataset):
        # (batch_size, input_dim(c), T, num_nodes)
        x_t1 = self.tconv1(x)
        # (batch_size, input_dim(c), 1, num_nodes)
        if self.mode == 'pretrain':
            x_t1 = self.ln_pretrain[self.dataset2index[select_dataset]](x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            x_t1 = self.ln_eval[self.dataset2index[select_dataset]](x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # x_t1 = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_ln = x_t1.permute(0, 2, 3, 1).permute(0, 3, 1, 2)
        # (batch_size, input_dim(c), 1, num_nodes)
        x_t2 = self.tconv2(x_ln)
        return self.fc_pretrain(x_t2)


class STGCN(nn.Module):
    def __init__(self, args, G_dict, dataset_use, dataset_test, num_nodes, dim_in, dim_out, device, mode):
        super(STGCN, self).__init__()
        self.Ks = args.Ks
        self.Kt = args.Kt
        self.num_nodes = num_nodes
        self.G_dict = G_dict
        self.blocks0 = [dim_in, args.blocks1[1], args.blocks1[0]]
        self.blocks1 = args.blocks1
        self.drop_prob = args.drop_prob
        self.device = device
        self.st_conv1 = STConvBlock(self.Ks, self.Kt, self.num_nodes,
                                    self.blocks0, self.drop_prob, self.G_dict, dataset_use, dataset_test, self.device, mode)
        self.st_conv2 = STConvBlock(self.Ks, self.Kt, self.num_nodes,
                                    self.blocks1, self.drop_prob, self.G_dict, dataset_use, dataset_test, self.device, mode)
        self.output = OutputLayer(args.blocks1[2], args.outputl_ks, self.num_nodes, dim_out, self.G_dict, dataset_use, dataset_test, mode)

    def forward(self, x, select_dataset):
        # print(x.shape)
        x = x.permute(0, 3, 1, 2)  # (batch_size, feature_dim, input_length, num_nodes)
        # print(x.shape)
        x_st1 = self.st_conv1(x, select_dataset)   # (batch_size, c[2](64), input_length-kt+1-kt+1, num_nodes)
        # print(x_st1.shape)
        x_st2 = self.st_conv2(x_st1, select_dataset)  # (batch_size, c[2](128), input_length-kt+1-kt+1-kt+1-kt+1, num_nodes)
        # print(x_st2.shape)
        outputs1 = self.output(x_st2, select_dataset)  # (batch_size, output_dim(1), output_length(1), num_nodes)
        # print(outputs.shape)
        outputs2 = outputs1.permute(0, 2, 3, 1)  # (batch_size, output_length(1), num_nodes, output_dim)
        # print(outputs2.shape)
        return outputs2