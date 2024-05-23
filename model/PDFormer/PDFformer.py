import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial
from logging import getLogger


def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, norm_layer=None):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.token_embed(x)
        x = self.norm(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].unsqueeze(2).expand_as(x).detach()


class LaplacianPE(nn.Module):
    def __init__(self, lape_dim, embed_dim):
        super().__init__()
        self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)

    def forward(self, lap_mx):
        lap_pos_enc = self.embedding_lap_pos_enc(lap_mx).unsqueeze(0).unsqueeze(0)
        return lap_pos_enc


class DataEmbedding(nn.Module):
    def __init__(
        self, feature_dim, embed_dim, lape_dim, adj_mx, drop=0.,
        add_time_in_day=False, add_day_in_week=False, device=torch.device('cpu'),
    ):
        super().__init__()

        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week

        self.device = device
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.value_embedding = TokenEmbedding(feature_dim, embed_dim)

        self.position_encoding = PositionalEncoding(embed_dim)
        if self.add_time_in_day:
            self.minute_size = 1440
            self.daytime_embedding = nn.Embedding(self.minute_size, embed_dim)
        if self.add_day_in_week:
            weekday_size = 7
            self.weekday_embedding = nn.Embedding(weekday_size, embed_dim)
        self.spatial_embedding = LaplacianPE(lape_dim, embed_dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x, lap_mx):
        origin_x = x
        x = self.value_embedding(origin_x[:, :, :, :self.feature_dim])
        x += self.position_encoding(x)
        if self.add_time_in_day:
            x += self.daytime_embedding((origin_x[:, :, :, self.feature_dim] * self.minute_size).round().long())
        if self.add_day_in_week:
            x += self.weekday_embedding(origin_x[:, :, :, self.feature_dim + 1: self.feature_dim + 8].argmax(dim=3))
        x += self.spatial_embedding(lap_mx)
        x = self.dropout(x)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :x.shape[2] - self.chomp_size, :].contiguous()


class STSelfAttention(nn.Module):
    def __init__(
        self, dim, s_attn_size, t_attn_size, geo_num_heads=4, sem_num_heads=2, t_num_heads=2, qkv_bias=False,
        attn_drop=0., proj_drop=0., device=torch.device('cpu'), output_dim=1,
    ):
        super().__init__()
        assert dim % (geo_num_heads + sem_num_heads + t_num_heads) == 0
        self.geo_num_heads = geo_num_heads
        self.sem_num_heads = sem_num_heads
        self.t_num_heads = t_num_heads
        self.head_dim = dim // (geo_num_heads + sem_num_heads + t_num_heads)
        self.scale = self.head_dim ** -0.5
        self.device = device
        self.s_attn_size = s_attn_size
        self.t_attn_size = t_attn_size
        self.geo_ratio = geo_num_heads / (geo_num_heads + sem_num_heads + t_num_heads)
        self.sem_ratio = sem_num_heads / (geo_num_heads + sem_num_heads + t_num_heads)
        self.t_ratio = 1 - self.geo_ratio - self.sem_ratio
        self.output_dim = output_dim

        self.pattern_q_linears = nn.ModuleList([
            nn.Linear(dim, int(dim * self.geo_ratio)) for _ in range(output_dim)
        ])
        self.pattern_k_linears = nn.ModuleList([
            nn.Linear(dim, int(dim * self.geo_ratio)) for _ in range(output_dim)
        ])
        self.pattern_v_linears = nn.ModuleList([
            nn.Linear(dim, int(dim * self.geo_ratio)) for _ in range(output_dim)
        ])

        self.geo_q_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias)
        self.geo_k_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias)
        self.geo_v_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias)
        self.geo_attn_drop = nn.Dropout(attn_drop)

        self.sem_q_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias)
        self.sem_k_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias)
        self.sem_v_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias)
        self.sem_attn_drop = nn.Dropout(attn_drop)

        self.t_q_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_k_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_v_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, x_patterns, pattern_keys, geo_mask=None, sem_mask=None):
        B, T, N, D = x.shape
        t_q = self.t_q_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_k = self.t_k_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_v = self.t_v_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_q = t_q.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_k = t_k.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_v = t_v.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale
        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.t_attn_drop(t_attn)
        t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, N, T, int(D * self.t_ratio)).transpose(1, 2)

        geo_q = self.geo_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_k = self.geo_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        for i in range(self.output_dim):
            pattern_q = self.pattern_q_linears[i](x_patterns[..., i])
            pattern_k = self.pattern_k_linears[i](pattern_keys[..., i])
            pattern_v = self.pattern_v_linears[i](pattern_keys[..., i])
            pattern_attn = (pattern_q @ pattern_k.transpose(-2, -1)) * self.scale
            pattern_attn = pattern_attn.softmax(dim=-1)
            geo_k += pattern_attn @ pattern_v
        geo_v = self.geo_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_q = geo_q.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_k = geo_k.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_v = geo_v.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_attn = (geo_q @ geo_k.transpose(-2, -1)) * self.scale
        if geo_mask is not None:
            geo_attn.masked_fill_(geo_mask, float('-inf'))
        geo_attn = geo_attn.softmax(dim=-1)
        geo_attn = self.geo_attn_drop(geo_attn)
        geo_x = (geo_attn @ geo_v).transpose(2, 3).reshape(B, T, N, int(D * self.geo_ratio))

        sem_q = self.sem_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sem_k = self.sem_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sem_v = self.sem_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sem_q = sem_q.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_k = sem_k.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_v = sem_v.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_attn = (sem_q @ sem_k.transpose(-2, -1)) * self.scale
        if sem_mask is not None:
            sem_attn.masked_fill_(sem_mask, float('-inf'))
        sem_attn = sem_attn.softmax(dim=-1)
        sem_attn = self.sem_attn_drop(sem_attn)
        sem_x = (sem_attn @ sem_v).transpose(2, 3).reshape(B, T, N, int(D * self.sem_ratio))

        x = self.proj(torch.cat([t_x, geo_x, sem_x], dim=-1))
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TemporalSelfAttention(nn.Module):
    def __init__(
        self, dim, dim_out, t_attn_size, t_num_heads=6, qkv_bias=False,
        attn_drop=0., proj_drop=0., device=torch.device('cpu'),
    ):
        super().__init__()
        assert dim % t_num_heads == 0
        self.t_num_heads = t_num_heads
        self.head_dim = dim // t_num_heads
        self.scale = self.head_dim ** -0.5
        self.device = device
        self.t_attn_size = t_attn_size

        self.t_q_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.t_k_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.t_v_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, N, D = x.shape
        t_q = self.t_q_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_k = self.t_k_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_v = self.t_v_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_q = t_q.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_k = t_k.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_v = t_v.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale

        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.t_attn_drop(t_attn)

        t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, N, T, D).transpose(1, 2)

        x = self.proj(t_x)
        x = self.proj_drop(x)
        return x


class STEncoderBlock(nn.Module):

    def __init__(
        self, dim, s_attn_size, t_attn_size, geo_num_heads=4, sem_num_heads=2, t_num_heads=2, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
        drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, device=torch.device('cpu'), type_ln="pre", output_dim=1,
    ):
        super().__init__()
        self.type_ln = type_ln
        self.norm1 = norm_layer(dim)
        self.st_attn = STSelfAttention(
            dim, s_attn_size, t_attn_size, geo_num_heads=geo_num_heads, sem_num_heads=sem_num_heads, t_num_heads=t_num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, device=device, output_dim=output_dim,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_patterns, pattern_keys, geo_mask=None, sem_mask=None):
        if self.type_ln == 'pre':
            x = x + self.drop_path(self.st_attn(self.norm1(x), x_patterns, pattern_keys, geo_mask=geo_mask, sem_mask=sem_mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif self.type_ln == 'post':
            x = self.norm1(x + self.drop_path(self.st_attn(x, x_patterns, pattern_keys, geo_mask=geo_mask, sem_mask=sem_mask)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x


class PDFormer(nn.Module):
    def __init__(self, args_predictor, args_base):
        super(PDFormer, self).__init__()

        self.num_nodes = args_base.num_nodes
        self.feature_dim = args_base.dim_in
        self.ext_dim = 0
        # self.dtw_matrix = args_predictor.dtw_matrix
        # self.adj_mx = args_predictor.adj_mx
        # self.lap_mx = args_predictor.lap_mx
        # sd_mx = args_predictor.sd_mx
        # sh_mx = args_predictor.sh_mx

        self.dtw_matrix_dict = args_predictor.dtw_matrix_dict
        self.adj_mx_dict = args_predictor.adj_mx_dict
        self.lap_mx_dict = args_predictor.lap_mx_dict
        sd_mx = args_predictor.sd_mx
        sh_mx_dict = args_predictor.sh_mx_dict

        self.embed_dim = args_predictor.embed_dim
        self.skip_dim = args_predictor.skip_dim
        lape_dim = args_predictor.lape_dim
        geo_num_heads = args_predictor.geo_num_heads
        sem_num_heads = args_predictor.sem_num_heads
        t_num_heads = args_predictor.t_num_heads
        mlp_ratio = args_predictor.mlp_ratio
        qkv_bias = args_predictor.qkv_bias
        drop = args_predictor.drop
        attn_drop = args_predictor.attn_drop
        drop_path = args_predictor.drop_path
        self.s_attn_size = args_predictor.s_attn_size
        self.t_attn_size = args_predictor.t_attn_size
        enc_depth = args_predictor.enc_depth
        type_ln = args_predictor.type_ln
        self.type_short_path = args_predictor.type_short_path

        self.output_dim = args_base.output_dim
        self.input_window = args_base.his
        self.output_window = args_base.pred
        add_time_in_day = args_predictor.add_time_in_day
        add_day_in_week = args_predictor.add_day_in_week
        self.device = args_base.device

        self.far_mask_delta = args_predictor.far_mask_delta
        self.dtw_delta = args_predictor.dtw_delta

        # self.use_curriculum_learning = config.get('use_curriculum_learning', True)
        # self.step_size = config.get('step_size', 2500)
        # self.task_level = config.get('task_level', 0)
        # if self.max_epoch * self.num_batches * self.world_size < self.step_size * self.output_window:
        #     self._logger.warning('Parameter `step_size` is too big with {} epochs and '
        #                          'the model cannot be trained for all time steps.'.format(self.max_epoch))
        # if self.use_curriculum_learning:
        #     self._logger.info('Use use_curriculum_learning!')

        if self.type_short_path == "dist":
            distances = sd_mx[~np.isinf(sd_mx)].flatten()
            std = distances.std()
            sd_mx = np.exp(-np.square(sd_mx / std))
            self.far_mask = torch.zeros(self.num_nodes, self.num_nodes).to(self.device)
            self.far_mask[sd_mx < self.far_mask_delta] = 1
            self.far_mask = self.far_mask.bool()
        else:
            # sh_mx = sh_mx.T
            # self.geo_mask = torch.zeros(self.num_nodes, self.num_nodes).to(self.device)
            # self.geo_mask[sh_mx >= self.far_mask_delta] = 1
            # self.geo_mask = self.geo_mask.bool()
            # self.sem_mask = torch.ones(self.num_nodes, self.num_nodes).to(self.device)
            # sem_mask = self.dtw_matrix.argsort(axis=1)[:, :self.dtw_delta]
            # for i in range(self.sem_mask.shape[0]):
            #     self.sem_mask[i][sem_mask[i]] = 0
            # self.sem_mask = self.sem_mask.bool()
            self.dataset2index = {}
            self.geo_mask = []
            self.sem_mask = []
            self.pattern_keys = []
            if args_base.mode == 'pretrain':
                for i, data_graph in enumerate(args_base.dataset_use):
                    self.dataset2index[data_graph] = i
                    sh_mx = sh_mx_dict[data_graph]
                    adj_mx = self.adj_mx_dict[data_graph]
                    num_nodes = adj_mx.shape[0]
                    sh_mx = sh_mx.T
                    self.geo_mask.append(torch.zeros(num_nodes, num_nodes).to(self.device))
                    self.geo_mask[i][sh_mx >= self.far_mask_delta] = 1
                    self.geo_mask[i] = self.geo_mask[i].bool()
                    self.sem_mask.append(torch.ones(num_nodes, num_nodes).to(self.device))
                    sem_mask = self.dtw_matrix_dict[data_graph].argsort(axis=1)[:, :self.dtw_delta]
                    for j in range(self.sem_mask[i].shape[0]):
                        self.sem_mask[i][j][sem_mask[j]] = 0
                    self.sem_mask[i] = self.sem_mask[i].bool()
                    self.pattern_keys.append(torch.from_numpy(args_predictor.pattern_key_matrix_dict[data_graph]).float().to(self.device))
                    self.enc_embed_layer = DataEmbedding(
                        self.feature_dim - self.ext_dim, self.embed_dim, lape_dim, self.adj_mx_dict[data_graph], drop=drop,
                        add_time_in_day=add_time_in_day, add_day_in_week=add_day_in_week, device=self.device,
                    )
            else:
                self.geo_mask = []
                self.sem_mask = []
                for i, data_graph in enumerate([args_base.dataset_test]):
                    self.dataset2index[data_graph] = i
                    sh_mx = sh_mx_dict[data_graph]
                    adj_mx = self.adj_mx_dict[data_graph]
                    num_nodes = adj_mx.shape[0]
                    sh_mx = sh_mx.T
                    self.geo_mask.append(torch.zeros(num_nodes, num_nodes).to(self.device))
                    self.geo_mask[i][sh_mx >= self.far_mask_delta] = 1
                    self.geo_mask[i] = self.geo_mask[i].bool()
                    self.sem_mask.append(torch.ones(num_nodes, num_nodes).to(self.device))
                    sem_mask = self.dtw_matrix_dict[data_graph].argsort(axis=1)[:, :self.dtw_delta]
                    for j in range(self.sem_mask[i].shape[0]):
                        self.sem_mask[i][j][sem_mask[j]] = 0
                    self.sem_mask[i] = self.sem_mask[i].bool()
                    self.pattern_keys.append(torch.from_numpy(args_predictor.pattern_key_matrix_dict[data_graph]).float().to(self.device))
                    self.enc_embed_layer = DataEmbedding(
                        self.feature_dim - self.ext_dim, self.embed_dim, lape_dim, self.adj_mx_dict[data_graph], drop=drop,
                        add_time_in_day=add_time_in_day, add_day_in_week=add_day_in_week, device=self.device,
                    )

        # self.pattern_keys = torch.from_numpy(args_predictor.pattern_key_matrix).float().to(self.device)
        self.pattern_embeddings = nn.ModuleList([
            TokenEmbedding(self.s_attn_size, self.embed_dim) for _ in range(self.output_dim)
        ])

        # self.enc_embed_layer = DataEmbedding(
        #     self.feature_dim - self.ext_dim, self.embed_dim, lape_dim, self.adj_mx_dict[], drop=drop,
        #     add_time_in_day=add_time_in_day, add_day_in_week=add_day_in_week, device=self.device,
        # )

        enc_dpr = [x.item() for x in torch.linspace(0, drop_path, enc_depth)]
        self.encoder_blocks = nn.ModuleList([
            STEncoderBlock(
                dim=self.embed_dim, s_attn_size=self.s_attn_size, t_attn_size=self.t_attn_size, geo_num_heads=geo_num_heads, sem_num_heads=sem_num_heads, t_num_heads=t_num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=enc_dpr[i], act_layer=nn.GELU,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), device=self.device, type_ln=type_ln, output_dim=self.output_dim,
            ) for i in range(enc_depth)
        ])

        self.skip_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=self.embed_dim, out_channels=self.skip_dim, kernel_size=1,
            ) for _ in range(enc_depth)
        ])

        self.end_conv1 = nn.Conv2d(
            in_channels=self.input_window, out_channels=self.output_window, kernel_size=1, bias=True,
        )
        self.end_conv2 = nn.Conv2d(
            in_channels=self.skip_dim, out_channels=self.output_dim, kernel_size=1, bias=True,
        )

    def forward(self, batch, select_dataset):
        x = batch
        T =  x.shape[1]
        x_pattern_list = []
        for i in range(self.s_attn_size):
            x_pattern = F.pad(
                x[:, :T + i + 1 - self.s_attn_size, :, :self.output_dim],
                (0, 0, 0, 0, self.s_attn_size - 1 - i, 0),
                "constant", 0,
            ).unsqueeze(-2)
            x_pattern_list.append(x_pattern)
        x_patterns = torch.cat(x_pattern_list, dim=-2)  # (B, T, N, s_attn_size, output_dim)

        x_pattern_list = []
        pattern_key_list = []
        for i in range(self.output_dim):
            x_pattern_list.append(self.pattern_embeddings[i](x_patterns[..., i]).unsqueeze(-1))
            pattern_key_list.append(self.pattern_embeddings[i](self.pattern_keys[self.dataset2index[select_dataset]][..., i]).unsqueeze(-1))
        x_patterns = torch.cat(x_pattern_list, dim=-1)
        pattern_keys = torch.cat(pattern_key_list, dim=-1)

        enc = self.enc_embed_layer(x, self.lap_mx_dict[select_dataset])
        skip = 0
        for i, encoder_block in enumerate(self.encoder_blocks):
            enc = encoder_block(enc, x_patterns, pattern_keys, self.geo_mask[self.dataset2index[select_dataset]], self.sem_mask[self.dataset2index[select_dataset]])
            skip += self.skip_convs[i](enc.permute(0, 3, 2, 1))

        skip = self.end_conv1(F.relu(skip.permute(0, 3, 2, 1)))
        skip = self.end_conv2(F.relu(skip.permute(0, 3, 2, 1)))
        return skip.permute(0, 3, 2, 1)

