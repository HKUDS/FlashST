import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .odegcn import ODEG


class Chomp1d(nn.Module):
    """
    extra dimension will be added by padding, remove it
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class TemporalConvNet(nn.Module):
    """
    time dilation convolution
    """

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        Args:
            num_inputs : channel's number of input data's feature
            num_channels : numbers of data feature tranform channels, the last is the output channel
            kernel_size : using 1d convolution, so the real kernel is (1, kernel_size)
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), dilation=(1, dilation_size),
                                  padding=(0, padding))
            self.conv.weight.data.normal_(0, 0.01)
            self.chomp = Chomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

            layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]

        self.network = nn.Sequential(*layers)
        self.downsample = nn.Conv2d(num_inputs, num_channels[-1], (1, 1)) if num_inputs != num_channels[-1] else None
        if self.downsample:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x, select_dataset=None):
        """
        like ResNet
        Args:
            X : input data of shape (B, N, T, F)
        """
        # permute shape to (B, F, N, T)
        y = x.permute(0, 3, 1, 2)
        y = F.relu(self.network(y) + self.downsample(y) if self.downsample else y)
        y = y.permute(0, 2, 3, 1)
        return y


class GCN(nn.Module):
    def __init__(self, A_hat, in_channels, out_channels, ):
        super(GCN, self).__init__()
        self.A_hat = A_hat
        self.theta = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.reset()

    def reset(self):
        stdv = 1. / math.sqrt(self.theta.shape[1])
        self.theta.data.uniform_(-stdv, stdv)

    def forward(self, X):
        y = torch.einsum('ij, kjlm-> kilm', self.A_hat, X)
        return F.relu(torch.einsum('kjlm, mn->kjln', y, self.theta))


class STGCNBlock(nn.Module):
    def __init__(self, time_steps, in_channels, out_channels, num_nodes, A_hat, A_dict, dataset_use, dataset_test, mode, device):
        """
        Args:
            in_channels: Number of input features at each node in each time step.
            out_channels: a list of feature channels in timeblock, the last is output feature channel
            num_nodes: Number of nodes in the graph
            A_hat: the normalized adjacency matrix
        """
        super(STGCNBlock, self).__init__()
        self.A_hat = A_hat
        self.temporal1 = TemporalConvNet(num_inputs=in_channels,
                                         num_channels=out_channels)
        self.odeg = ODEG(out_channels[-1], time_steps, A_hat, time=6, dataset_use=dataset_use, dataset_test=dataset_test, mode=mode, device=device)
        self.temporal2 = TemporalConvNet(num_inputs=out_channels[-1],
                                         num_channels=out_channels)

        self.dataset2index = {}
        self.mode = mode
        if mode == 'pretrain':
            self.bn_pretrain = nn.ModuleList()
            for i, data_graph in enumerate(dataset_use):
                self.dataset2index[data_graph] = i
                n_dataset = A_dict[data_graph].shape[0]
                self.bn_pretrain.append(nn.BatchNorm2d(n_dataset))
        else:
            self.bn_eval = nn.ModuleList()
            for i, data_graph in enumerate([dataset_test]):
                self.dataset2index[data_graph] = i
                n_dataset = A_dict[data_graph].shape[0]
                self.bn_eval.append(nn.BatchNorm2d(n_dataset))

        # self.batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(self, X, select_dataset):
        # """
        # Args:
        #     X: Input data of shape (batch_size, num_nodes, num_timesteps, num_features)
        # Return:
        #     Output data of shape(batch_size, num_nodes, num_timesteps, out_channels[-1])
        # """
        t = self.temporal1(X)
        t = self.odeg(t, select_dataset)
        t = self.temporal2(F.relu(t))

        if self.mode == 'pretrain':
            return self.bn_pretrain[self.dataset2index[select_dataset]](t)
        else:
            return self.bn_eval[self.dataset2index[select_dataset]](t)
        # return self.batch_norm(t)


class ODEGCN(nn.Module):
    """ the overall network framework """

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, A_sp_hat, A_se_hat, dim_out, A_dict, dataset_use, dataset_test, mode, device):
        """
        Args:
            num_nodes : number of nodes in the graph
            num_features : number of features at each node in each time step
            num_timesteps_input : number of past time steps fed into the network
            num_timesteps_output : desired number of future time steps output by the network
            A_sp_hat : nomarlized adjacency spatial matrix
            A_se_hat : nomarlized adjacency semantic matrix
        """

        super(ODEGCN, self).__init__()

        # spatial graph
        self.sp_blocks1 = nn.ModuleList(
            [
                STGCNBlock(num_timesteps_input, in_channels=num_features, out_channels=[64, 32, 64],
                           num_nodes=num_nodes, A_hat=A_sp_hat, A_dict=A_dict, dataset_use=dataset_use,
                           dataset_test=dataset_test, mode=mode, device=device) for _ in range(3)
            ])
        self.sp_blocks2 = nn.ModuleList(
            [
                STGCNBlock(num_timesteps_input, in_channels=64, out_channels=[64, 32, 64],
                           num_nodes=num_nodes, A_hat=A_sp_hat, A_dict=A_dict, dataset_use=dataset_use,
                           dataset_test=dataset_test, mode=mode, device=device) for _ in range(3)
            ])


        # self.sp_blocks = nn.ModuleList(
        #     [nn.Sequential(
        #         STGCNBlock(num_timesteps_input, in_channels=num_features, out_channels=[64, 32, 64],
        #                    num_nodes=num_nodes, A_hat=A_sp_hat, A_dict=A_dict, dataset_use=dataset_use,
        #                    dataset_test=dataset_test, mode=mode),
        #         STGCNBlock(num_timesteps_input, in_channels=64, out_channels=[64, 32, 64],
        #                    num_nodes=num_nodes, A_hat=A_sp_hat, A_dict=A_dict, dataset_use=dataset_use,
        #                    dataset_test=dataset_test, mode=mode)) for _ in range(3)
        #     ])

        # semantic graph
        self.se_blocks1 = nn.ModuleList([
            STGCNBlock(num_timesteps_input, in_channels=num_features, out_channels=[64, 32, 64],
                       num_nodes=num_nodes, A_hat=A_se_hat, A_dict=A_dict, dataset_use=dataset_use,
                           dataset_test=dataset_test, mode=mode, device=device) for _ in range(3)
        ])

        self.se_blocks2 = nn.ModuleList([
            STGCNBlock(num_timesteps_input, in_channels=64, out_channels=[64, 32, 64],
                       num_nodes=num_nodes, A_hat=A_se_hat, A_dict=A_dict, dataset_use=dataset_use,
                           dataset_test=dataset_test, mode=mode, device=device) for _ in range(3)
        ])

        # self.se_blocks = nn.ModuleList([nn.Sequential(
        #     STGCNBlock(num_timesteps_input, in_channels=num_features, out_channels=[64, 32, 64],
        #                num_nodes=num_nodes, A_hat=A_se_hat, A_dict=A_dict, dataset_use=dataset_use,
        #                    dataset_test=dataset_test, mode=mode),
        #     STGCNBlock(num_timesteps_input, in_channels=64, out_channels=[64, 32, 64],
        #                num_nodes=num_nodes, A_hat=A_se_hat, A_dict=A_dict, dataset_use=dataset_use,
        #                    dataset_test=dataset_test, mode=mode)) for _ in range(3)
        # ])

        self.pred = nn.Sequential(
            nn.Linear(num_timesteps_input * 64, num_timesteps_output * 32),
            nn.ReLU(),
            nn.Linear(num_timesteps_output * 32, num_timesteps_output * dim_out)
        )
        self.dim_out = dim_out

    def forward(self, x, select_dataset):
        # """
        # Args:
        #     x : input data of shape (batch_size, num_nodes, num_timesteps, num_features) == (B, N, T, F)
        # Returns:
        #     prediction for future of shape (batch_size, num_nodes, num_timesteps_output)
        # """
        x = x.transpose(1, 2)
        outs = []

        # spatial graph
        for blk1, blk2 in zip(self.sp_blocks1, self.sp_blocks2):
            x1 = blk1(x, select_dataset)
            outs.append(blk2(x1, select_dataset))
        # semantic graph
        for blk1, blk2 in zip(self.se_blocks1, self.se_blocks2):
            x1 = blk1(x, select_dataset)
            outs.append(blk2(x1, select_dataset))
        outs = torch.stack(outs)
        x = torch.max(outs, dim=0)[0]
        x = x.reshape((x.shape[0], x.shape[1], -1))
        if self.dim_out != 1:
            batch, node_num, time_step = x.shape[0], x.shape[1], x.shape[2]
            out_pred = self.pred(x).unsqueeze(-1).reshape(batch, node_num, -1, self.dim_out).transpose(1, 2)
        else:
            out_pred = self.pred(x).unsqueeze(-1).transpose(1, 2)
        # print(out_pred.shape)

        return out_pred
