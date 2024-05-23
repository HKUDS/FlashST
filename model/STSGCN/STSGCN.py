import torch
import torch.nn as nn
# from torch.nn.init import xavier_uniform
import numpy as np
from logging import getLogger


# 时空嵌入矩阵，真正的时空特征的嵌入表示
class PositionEmbedding(nn.Module):
    def __init__(self, input_length, num_of_vertices, embedding_size, A_dict, dataset_use, dataset_test, mode, temporal=True, spatial=True):
        super(PositionEmbedding, self).__init__()
        self.mode = mode
        self.input_length = input_length
        self.num_of_vertices = num_of_vertices
        self.embedding_size = embedding_size
        self.temporal = temporal
        self.spatial = spatial
        self.temporal_emb = torch.nn.Parameter(torch.zeros((1, input_length, 1, embedding_size)).to('cuda:0'))
        # shape is (1, T, 1, C)
        # xavier_uniform(self.temporal_emb)

        self.dataset2index = {}
        if mode == 'pretrain':
            self.spatial_emb_pretrain = nn.ParameterList()
            for i, data_graph in enumerate(dataset_use):
                self.dataset2index[data_graph] = i
                n_dataset = A_dict[data_graph].shape[0]
                self.spatial_emb_pretrain.append(torch.nn.Parameter(torch.zeros((1, 1, n_dataset, embedding_size)).to('cuda:0')))
        else:
            self.spatial_emb_eval = nn.ParameterList()
            for i, data_graph in enumerate([dataset_test]):
                self.dataset2index[data_graph] = i
                n_dataset = A_dict[data_graph].shape[0]
                self.spatial_emb_eval.append(torch.nn.Parameter(torch.zeros((1, 1, n_dataset, embedding_size)).to('cuda:0')))


        # self.spatial_emb = torch.nn.Parameter(torch.zeros((1, 1, num_of_vertices, embedding_size)).to('cuda:0'))
        # shape is (1, 1, N, C)
        # xavier_uniform(self.spatial_emb)

    def forward(self, data, select_dataset):
        if self.temporal:
            data += self.temporal_emb
        if self.spatial:
            if self.mode == 'pretrain':
                data += self.spatial_emb_pretrain[self.dataset2index[select_dataset]]
            else:
                data += self.spatial_emb_eval[self.dataset2index[select_dataset]]
        return data


# 图卷积，没啥好说的
class GcnOperation(nn.Module):
    def __init__(self, num_of_filter, num_of_features, num_of_vertices, activation):
        super().__init__()
        assert activation in {'GLU', 'relu'}

        self.num_of_filter = num_of_filter
        self.num_of_features = num_of_features
        self.num_of_vertices = num_of_vertices
        self.activation = activation
        if activation == "GLU":
            self.layer = nn.Linear(num_of_features, 2 * num_of_filter)
        elif activation == "relu":
            self.layer = nn.Linear(num_of_features, num_of_filter)

    def forward(self, data, adj):
        data = torch.matmul(adj, data)

        if self.activation == "GLU":
            data = self.layer(data)
            lhs, rhs = data.split(self.num_of_filter, -1)
            data = lhs * torch.sigmoid(rhs)

        elif self.activation == "relu":
            data = torch.relu(self.layer(data))

        return data


# 同步时空卷积模块，捕获连续 3 个时间片的时空特征
class Stsgcm(nn.Module):
    def __init__(self, filters, num_of_features, num_of_vertices, activation):
        super().__init__()
        self.filters = filters
        self.num_of_features = num_of_features
        self.num_of_vertices = num_of_vertices
        self.activation = activation
        self.layers = nn.ModuleList()
        for i in range(len(filters)):
            self.layers.append(GcnOperation(filters[i], num_of_features, num_of_vertices, activation))
            num_of_features = filters[i]

    def forward(self, data, adj):
        # 多个卷积层叠加，
        num_nodes = adj.shape[0] // 3
        need_concat = []
        for i in range(len(self.layers)):
            data = self.layers[i](data, adj)
            need_concat.append(torch.transpose(data, 1, 0))

        # 且每个卷积层的输出都以类似残差网络的形式输入聚合层
        need_concat = [
            torch.unsqueeze(
                i[num_nodes:2 * num_nodes, :, :],
                dim=0
            ) for i in need_concat
        ]

        # 聚合使用最大池化
        return torch.max(torch.cat(need_concat, dim=0), dim=0)[0]


class STSGCL(nn.Module):
    def __init__(self, t, num_of_vertices, num_of_features, filters, module_type, activation, A_dict,
                 dataset_use, dataset_test, mode, temporal_emb=True, spatial_emb=True):
        super().__init__()
        assert module_type in {'sharing', 'individual'}
        self.T = t
        self.num_of_vertices = num_of_vertices
        self.num_of_features = num_of_features
        self.filters = filters
        self.module_type = module_type
        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb

        if module_type == 'individual':
            self.layer = STSGCNLayerIndividual(
                t, num_of_vertices, num_of_features, filters,
                activation, A_dict, dataset_use, dataset_test, mode, temporal_emb, spatial_emb
            )
        else:
            self.layer = STSGCNLayerSharing(
                t, num_of_vertices, num_of_features, filters,
                activation, A_dict, dataset_use, dataset_test, mode, temporal_emb, spatial_emb
            )

    def forward(self, data, adj, select_dataset):
        return self.layer(data, adj, select_dataset)


class STSGCNLayerIndividual(nn.Module):
    def __init__(self, t, num_of_vertices, num_of_features, filters,
                 activation,  A_dict, dataset_use, dataset_test, mode, temporal_emb=True, spatial_emb=True):
        super().__init__()
        self.T = t
        self.num_of_vertices = num_of_vertices
        self.num_of_features = num_of_features
        self.filters = filters
        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb
        self.position_embedding = PositionEmbedding(t, num_of_vertices, num_of_features, A_dict, dataset_use, dataset_test, mode,
                                                    temporal_emb, spatial_emb)

        # 一个 GCM 模块可以捕获连续 3 个时间片的时空特征
        # T 个时间片，每 3 个一捕获，即 T - 2 次，对应 T - 2 个 GCM 模块
        self.gcms = nn.ModuleList()
        for i in range(self.T - 2):
            self.gcms.append(Stsgcm(self.filters, self.num_of_features, self.num_of_vertices,
                                    activation=self.activation))

    def forward(self, data, adj, select_dataset):
        data = self.position_embedding(data, select_dataset)
        need_concat = []
        num_nodes = data.shape[2]

        for i in range(self.T - 2):
            t = data[:, i:i + 3, :, :]
            # shape is (B, 3, N, C)

            t = torch.reshape(t, (-1, 3 * num_nodes, self.num_of_features))
            # shape is (B, 3N, C)

            t = self.gcms[i](t, adj)
            # shape is (N, B, C')

            t = torch.transpose(t, 0, 1)
            # shape is (B, N, C')

            need_concat.append(torch.unsqueeze(t, dim=1))
            # shape is (B, 1, N, C')

        # 拼接各个 GCM 模块的输出
        return torch.cat(need_concat, dim=1)
        # shape is (B, T-2, N, C')


# 论文作者消融实验使用，与 Sthgcn_layer_individual 的区别在于，共用一个 stsgcm 模块
class STSGCNLayerSharing(nn.Module):
    def __init__(self, t, num_of_vertices, num_of_features, filters,
                 activation,  A_dict, dataset_use, dataset_test, mode, temporal_emb=True, spatial_emb=True):
        super().__init__()
        self.T = t
        self.num_of_vertices = num_of_vertices
        self.num_of_features = num_of_features
        self.filters = filters
        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb
        self.position_embedding = PositionEmbedding(t, num_of_vertices, num_of_features,  A_dict, dataset_use, dataset_test, mode,
                                                    temporal_emb, spatial_emb)
        self.gcm = Stsgcm(self.filters, self.num_of_features, self.num_of_vertices,
                          activation=self.activation)

    def forward(self, data, adj, select_dataset):
        data = self.position_embedding(data, select_dataset)
        num_nodes = data.shape[2]
        need_concat = []
        for i in range(self.T - 2):
            t = data[:, i:i + 3, :, :]
            # shape is (B, 3, N, C)

            t = torch.reshape(t, (-1, 3 * num_nodes, self.num_of_features))
            # shape is (B, 3N, C)

            need_concat.append(t)
            # shape is (B, 3N, C)

        t = torch.cat(need_concat, dim=0)
        # shape is ((T-2)*B, 3N, C)

        t = self.gcm(t, adj)
        # shape is (N, (T-2)*B, C')



        t = t.reshape((num_nodes, self.T - 2, -1, self.filters[-1]))
        # shape is (N, T - 2, B, C)

        return torch.transpose(t, 0, 2)
        # shape is (B, T - 2, N, C)


class OutputLayer(nn.Module):
    def __init__(self, num_of_vertices, input_length, num_of_features,
                 num_of_filters=128, predict_length=12, output_dim=1):
        super().__init__()
        self.num_of_vertices = num_of_vertices
        self.input_length = input_length
        self.num_of_features = num_of_features
        self.num_of_filters = num_of_filters
        self.predict_length = predict_length
        self.output_dim = output_dim
        self.hidden_layer = nn.Linear(self.input_length * self.num_of_features, self.num_of_filters)
        self.ouput_layer = nn.Linear(self.num_of_filters, self.predict_length * self.output_dim)

    def forward(self, data):
        batch_size = data.shape[0]
        data = torch.transpose(data, 1, 2)

        # (B, N, T * C)
        data = torch.reshape(
            data, (batch_size, -1, self.input_length * self.num_of_features)
        )

        # (B, N, C')
        # data = self.hidden_layer(data)
        # data = torch.relu(self.hidden_layer(data))
        data = self.hidden_layer(data)

        # (B, N, T' * C_out) -> (B, N, T', C_out)
        data = self.ouput_layer(data).reshape(
            (batch_size, -1, self.predict_length, self.output_dim)
        )

        # (B, T', N, C_out)
        data = data.permute(0, 2, 1, 3)

        return data


def construct_adj(a, steps):
    """
    构造局部时空图

    Parameters
    ----------
    a: np.ndarray, shape： (N, N), 原图邻接矩阵

    steps: 时间步长度，原论文是 3 个一合

    Returns
    ----------
    局部时空图矩阵 shape：(N * steps, N * steps)
    """
    n = len(a)
    adj = np.zeros([n * steps] * 2)

    for i in range(steps):
        adj[i * n: (i + 1) * n, i * n: (i + 1) * n] = a

    # 实际就是加了相邻两个时间步节点到自身的边
    for i in range(n):
        for k in range(steps - 1):
            adj[k * n + i, (k + 1) * n + i] = 1
            adj[(k + 1) * n + i, k * n + i] = 1

    for i in range(len(adj)):
        adj[i, i] = 1

    return adj


class STSGCN(nn.Module):
    def __init__(self, args, num_nodes, time_steps, dim_in, dim_out, A_dict, dataset_use, dataset_test, device, mode):
        super(STSGCN, self).__init__()
        self.num_nodes = num_nodes
        self.feature_dim = args.feature_dim
        self.output_dim = dim_out
        self._logger = getLogger()

        # 加载配置信息
        self.module_type = args.module_type
        self.activation = args.activation
        self.temporal_emb = args.temporal_emb
        self.spatial_emb = args.spatial_emb
        self.use_mask = args.use_mask
        self.device = device
        self.input_window = args.input_window
        self.output_window = args.output_window
        self.rho = args.rho
        self.num_of_vertices = self.num_nodes
        self.input_length = self.input_window
        self.predict_length = self.output_window


        # 构造局部时空图
        self.adj = {}
        self.mode = mode
        if self.mode == 'pretrain':
            for data_graph in dataset_use:
                self.adj[data_graph] = construct_adj(A_dict[data_graph].cpu(), 3)
                self.adj[data_graph] = torch.tensor(self.adj[data_graph], requires_grad=False, dtype=torch.float32).to(self.device)
        else:
            for data_graph in [dataset_test]:
                self.adj[data_graph] = construct_adj(A_dict[data_graph].cpu(), 3)
                self.adj[data_graph] = torch.tensor(self.adj[data_graph], requires_grad=False, dtype=torch.float32).to(self.device)

        # self.adj = A
        # self.adj = construct_adj(self.adj, 3)
        # self.adj = torch.tensor(self.adj, requires_grad=False, dtype=torch.float32).to(self.device)



        if self.use_mask:
            # 初始化遮罩矩阵
            self._logger.warning('You use mask matrix, please make sure you set '
                                 '`loss.backward(retain_graph=True)` to replace the '
                                 '`loss.backward()` in traffic_state_executor.py!')
            self.mask = torch.nn.Parameter(torch.tensor((self.adj != 0) + 0.0).to(self.device))
            self.adj = self.mask * self.adj

        # 输入嵌入层，增加特征数量，提升表示能力
        self.embedding_dim = self.feature_dim
        self.num_of_features = dim_in
        first_layer_embedding_size = args.first_layer_embedding_size
        if first_layer_embedding_size:
            self.first_layer_embedding = nn.Linear(self.num_of_features, first_layer_embedding_size)
            self.num_of_features = first_layer_embedding_size
        else:
            self.first_layer_embedding = None

        # 时空同步卷积层组
        self.filter_list = args.filter_list
        self.stsgcl_layers = nn.ModuleList()
        for idx, filters in enumerate(self.filter_list):
            # if self.input_length <= 0:
                # break
            self.stsgcl_layers.append(STSGCL(self.input_length, self.num_of_vertices,
                                             self.num_of_features, filters, self.module_type,
                                             self.activation, A_dict, dataset_use, dataset_test, mode,
                                             temporal_emb=self.temporal_emb,
                                             spatial_emb=self.spatial_emb))
            self.input_length -= 2
            self.num_of_features = filters[-1]

        # 输出层，每个预测时间步一个全连接层
        self.outputs = nn.ModuleList()
        for i in range(self.predict_length):
            self.outputs.append(OutputLayer(self.num_of_vertices, self.input_length, self.num_of_features,
                                            num_of_filters=128, predict_length=1, output_dim=self.output_dim))

    def forward(self, source, select_dataset):
        data = source
        # data.shape = (B:batch_size, T:input_length, N:vertical_num, C:feature_num)
        # if data.shape[-1] > self.embedding_dim:
        #     data = data[:, :, :, 0:self.embedding_dim]

        # print('data_1', data.shape)

        # data.shape = (B, T, N, C:embedding_feature_num)
        if self.first_layer_embedding:
            data = torch.relu(self.first_layer_embedding(data))
        # print('data_1', data.shape)

        for stsgcl_layer in self.stsgcl_layers:
            data = stsgcl_layer(data.clone(), self.adj[select_dataset], select_dataset)
        # print('data_2', data.shape)

        need_concat = []
        for output_layer in self.outputs:
            need_concat.append(output_layer(data))

        data = torch.cat(need_concat, dim=1)
        # print('data_3', data)

        return data
