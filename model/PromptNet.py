import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class GCN(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels=hidden_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.LeakyReLU()

    def forward(self, input_data: torch.Tensor, nadj: torch.Tensor, useGNN=False) -> torch.Tensor:
        if useGNN:
            gcn_out = self.act(torch.einsum('nk,bdke->bdne', nadj, self.fc1(input_data)))
            # gcn_out = self.act(self.fc1(torch.einsum('nk,bdke->bdne', nadj, input_data)))
        else:
            gcn_out = self.act(self.fc1(input_data))
        return gcn_out + input_data


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))
        hidden = hidden + input_data
        return hidden

class PromptNet(nn.Module):
    def __init__(self, args):
        super(PromptNet, self).__init__()

        self.mode = args.mode
        self.node_dim = args.node_dim
        self.input_len = args.his
        self.embed_dim = args.embed_dim
        self.output_len = args.pred
        self.num_layer = args.num_layer
        self.temp_dim_tid = args.temp_dim_tid
        self.temp_dim_diw = args.temp_dim_diw

        self.if_time_in_day = args.if_time_in_day
        self.if_day_in_week = args.if_day_in_week
        self.if_spatial = args.if_spatial

        self.input_base_dim = args.input_base_dim
        self.input_extra_dim = args.input_extra_dim
        self.data_type = args.data_type

        # spatial embeddings
        if self.if_spatial:
            self.LaplacianPE1 = nn.Linear(self.node_dim, self.node_dim)
            self.LaplacianPE2 = nn.Linear(self.node_dim, self.node_dim)

        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Embedding(288+1, self.temp_dim_tid)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Embedding(7+1, self.temp_dim_diw)

        self.time_series_emb_layer = nn.Linear(self.input_len, self.embed_dim, bias=True)

        self.hidden_dim = self.embed_dim+self.node_dim * \
            int(self.if_spatial)+self.temp_dim_tid*int(self.if_day_in_week) + \
            self.temp_dim_diw*int(self.if_time_in_day)

        # Base
        self.encoder1 = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])
        self.encoder2 = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])
        self.gcn1 = GCN(self.hidden_dim)
        self.gcn2 = GCN(self.hidden_dim)

        self.act = nn.LeakyReLU()

    def forward(self, history_data, source2, batch_seen=None, nadj=None, lpls=None, useGNN=False, DSU=True):

        input_data = history_data
        batch_size, _, num_nodes, _ = input_data.shape

        ZERO = torch.IntTensor(1).to('cuda:0')
        if self.if_time_in_day:
            t_i_d_data = source2[:, 0, :, self.input_base_dim]
            time_in_day_emb = self.time_in_day_emb(t_i_d_data[:, :].type_as(ZERO))
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            d_i_w_data = source2[:, 0, :, self.input_base_dim+1]
            day_in_week_emb = self.day_in_week_emb(d_i_w_data[:, :].type_as(ZERO))
        else:
            day_in_week_emb = None

        time_series_emb = self.time_series_emb_layer(input_data[..., 0:self.input_base_dim].transpose(1, 3))

        node_emb = []
        if self.if_spatial:
            lap_pos_enc = self.LaplacianPE2(self.act(self.LaplacianPE1(lpls)))
            tensor_neb = lap_pos_enc.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(1).repeat(1, self.input_base_dim, 1, 1)
            node_emb.append(tensor_neb)

        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.unsqueeze(1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.unsqueeze(1))

        # concate all embeddings
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=-1).transpose(1, 3)

        # encoding
        hidden_gcn = self.gcn1(hidden, nadj, useGNN)
        hidden = self.encoder1(hidden_gcn)
        hidden_gcn = self.gcn2(hidden, nadj, useGNN)
        hidden = self.encoder2(hidden_gcn)
        x_prompt = hidden.transpose(1, 3) + input_data[..., 0:self.input_base_dim]
        x_prompt = F.normalize(x_prompt, dim=-1)
        return x_prompt

