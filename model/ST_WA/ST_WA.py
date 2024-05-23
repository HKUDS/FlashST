import torch
import torch.nn as nn
from .attention import TemporalAttention, SpatialAttention
# from util import reparameterize

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

class STWA(nn.Module):
    def __init__(self, device, num_nodes, input_dim, output_dim, channels, dynamic, lag, horizon, supports,
                 memory_size, A_dict, dataset_use, dataset_test, mode):
        super(STWA, self).__init__()
        self.supports = supports
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.channels = channels
        self.dynamic = dynamic
        self.horizon = horizon
        self.start_fc = nn.Linear(in_features=input_dim, out_features=self.channels)
        self.memory_size = memory_size

        if input_dim != 1:
            self.eval_dimin = nn.Linear(in_features=input_dim, out_features=1)

        self.layers = nn.ModuleList(
            [
                Layer(device=device, input_dim=channels, dynamic=dynamic, num_nodes=num_nodes, cuts=12,
                      cut_size=6, no_proxies=2, memory_size=memory_size, A_dict=A_dict, dataset_use=dataset_use, dataset_test=dataset_test, mode=mode),
                Layer(device=device, input_dim=channels, dynamic=dynamic, num_nodes=num_nodes, cuts=3,
                      cut_size=4, no_proxies=2, memory_size=memory_size, A_dict=A_dict, dataset_use=dataset_use, dataset_test=dataset_test, mode=mode),
                Layer(device=device, input_dim=channels, dynamic=dynamic, num_nodes=num_nodes, cuts=1,
                      cut_size=3, no_proxies=2, memory_size=memory_size, A_dict=A_dict, dataset_use=dataset_use, dataset_test=dataset_test, mode=mode),
            ])

        self.skip_layers = nn.ModuleList([
            nn.Linear(in_features=12 * channels, out_features=256),
            nn.Linear(in_features=3 * channels, out_features=256),
            nn.Linear(in_features=1 *channels, out_features=256),
        ])

        self.projections = nn.Sequential(*[
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, horizon * self.output_dim)])

        if self.dynamic:
            self.mu_estimator = nn.Sequential(*[
                nn.Linear(lag, 32),
                nn.Tanh(),
                nn.Linear(32, 32),
                nn.Tanh(),
                nn.Linear(32, memory_size)
            ])

            self.logvar_estimator = nn.Sequential(*[
                nn.Linear(lag, 32),
                nn.Tanh(),
                nn.Linear(32, 32),
                nn.Tanh(),
                nn.Linear(32, memory_size)
            ])

    def forward(self, x, select_dataset):
        if self.dynamic:
            if x.shape[-1] != 1:
                x_dm = self.eval_dimin(x)
            else:
                x_dm = x
            mu = self.mu_estimator(x_dm.transpose(3, 1).squeeze(1))
            logvar = self.logvar_estimator(x_dm.transpose(3, 1).squeeze(1))
            z_data = reparameterize(mu, logvar)
        else:
            z_data = 0


        x = self.start_fc(x)
        batch_size = x.size(0)
        num_nodes = x.shape[2]

        skip = 0
        for layer, skip_layer in zip(self.layers, self.skip_layers):
            x = layer(x, z_data, select_dataset)
            skip_inp = x.transpose(2, 1).reshape(batch_size, num_nodes, -1)
            skip = skip + skip_layer(skip_inp)

        x = torch.relu(skip)
        out = self.projections(x)

        if self.output_dim == 1:
            out = out.transpose(2, 1).unsqueeze(-1)
        else:
            out = out.unsqueeze(-1).reshape(batch_size, num_nodes, self.horizon, -1).transpose(2, 1)

        # print(out.shape)

        return out


class Layer(nn.Module):
    def __init__(self, device, input_dim, num_nodes, cuts, cut_size, dynamic, memory_size, no_proxies, A_dict, dataset_use, dataset_test, mode):
        super(Layer, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.dynamic = dynamic
        self.cuts = cuts
        self.cut_size = cut_size
        self.no_proxies = no_proxies
        self.mode = mode

        self.dataset2index = {}
        if mode == 'pretrain':
            self.proxies_pretrain = nn.ParameterList()
            self.mu_pretrain = nn.ParameterList()
            self.logvar_pretrain = nn.ParameterList()
            for i, data_graph in enumerate(dataset_use):
                self.dataset2index[data_graph] = i
                n_dataset = A_dict[data_graph].shape[0]
                self.proxies_pretrain.append(nn.Parameter(torch.randn(1, cuts * no_proxies, n_dataset, input_dim).to(device),
                                    requires_grad=True))
                if self.dynamic:
                    self.mu_pretrain.append(nn.Parameter(torch.randn(n_dataset, memory_size).to(device), requires_grad=True).to(
                        device))
                    self.logvar_pretrain.append(nn.Parameter(torch.randn(n_dataset, memory_size).to(device), requires_grad=True).to(
                        device))
        else:
            self.proxies_eval = nn.ParameterList()
            self.mu_eval = nn.ParameterList()
            self.logvar_eval = nn.ParameterList()
            for i, data_graph in enumerate([dataset_test]):
                self.dataset2index[data_graph] = i
                n_dataset = A_dict[data_graph].shape[0]
                self.proxies_eval.append(nn.Parameter(torch.randn(1, cuts * no_proxies, n_dataset, input_dim).to(device),
                                    requires_grad=True).to(device))

                if self.dynamic:
                    self.mu_eval.append(nn.Parameter(torch.randn(n_dataset, memory_size).to(device), requires_grad=True).to(
                        device))
                    self.logvar_eval.append(nn.Parameter(torch.randn(n_dataset, memory_size).to(device), requires_grad=True).to(
                        device))

        # self.proxies = nn.Parameter(torch.randn(1, cuts * no_proxies, self.num_nodes, input_dim).to(device),
        #                             requires_grad=True).to(device)

        self.temporal_att = TemporalAttention(input_dim, num_nodes=num_nodes, cut_size=cut_size)
        self.spatial_att = SpatialAttention(input_dim, num_nodes=num_nodes)

        # if self.dynamic:
        #     self.mu = nn.Parameter(torch.randn(num_nodes, memory_size).to(device), requires_grad=True).to(device)
        #     self.logvar = nn.Parameter(torch.randn(num_nodes, memory_size).to(device), requires_grad=True).to(device)
        #
        self.temporal_parameter_generators = nn.ModuleList([
            ParameterGenerator(memory_size=memory_size, input_dim=input_dim, output_dim=input_dim,
                               num_nodes=num_nodes, dynamic=dynamic) for _ in range(2)
        ])

        self.spatial_parameter_generators = nn.ModuleList([
            ParameterGenerator(memory_size=memory_size, input_dim=input_dim, output_dim=input_dim,
                               num_nodes=num_nodes, dynamic=dynamic) for _ in range(2)
        ])

        self.aggregator = nn.Sequential(*[
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        ])

    def forward(self, x, z_data, select_dataset):
        # x shape: B T N C
        batch_size = x.size(0)

        if self.dynamic:
            if self.mode == 'pretrain':
                z_sample = reparameterize(self.mu_pretrain[self.dataset2index[select_dataset]], self.logvar_pretrain[self.dataset2index[select_dataset]])
            else:
                z_sample = reparameterize(self.mu_eval[self.dataset2index[select_dataset]], self.logvar_eval[self.dataset2index[select_dataset]])
            # z_sample = reparameterize(self.mu, self.logvar)
            z_data = z_data + z_sample

        temporal_parameters = [layer(x, z_data) for layer in self.temporal_parameter_generators]
        spatial_parameters = [layer(x, z_data) for layer in self.spatial_parameter_generators]

        data_concat = []
        out = 0
        for i in range(self.cuts):
            # shape is (B, cut_size, N, C)
            t = x[:, i * self.cut_size:(i + 1) * self.cut_size, :, :]

            if self.mode == 'pretrain':
                proxies = self.proxies_pretrain[self.dataset2index[select_dataset]][:, i * self.no_proxies: (i + 1) * self.no_proxies]
            else:
                proxies = self.proxies_eval[self.dataset2index[select_dataset]][:, i * self.no_proxies: (i + 1) * self.no_proxies]
            # proxies = self.proxies[:, i * self.no_proxies: (i + 1) * self.no_proxies]
            proxies = proxies.repeat(batch_size, 1, 1, 1) + out
            t = torch.cat([proxies, t], dim=1)

            out = self.temporal_att(t[:, :self.no_proxies, :, :], t, t, temporal_parameters)
            out = self.spatial_att(out, spatial_parameters)
            out = (self.aggregator(out) * out).sum(1, keepdim=True)
            data_concat.append(out)

        return torch.cat(data_concat, dim=1)

class ParameterGenerator(nn.Module):
    def __init__(self, memory_size, input_dim, output_dim, num_nodes, dynamic):
        super(ParameterGenerator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.dynamic = dynamic

        if self.dynamic:
            print('Using DYNAMIC')
            self.weight_generator = nn.Sequential(*[
                nn.Linear(memory_size, 32),
                nn.ReLU(),
                nn.Linear(32, 5),
                nn.ReLU(),
                nn.Linear(5, input_dim * output_dim)
            ])
            self.bias_generator = nn.Sequential(*[
                nn.Linear(memory_size, 32),
                nn.ReLU(),
                nn.Linear(32, 5),
                nn.ReLU(),
                nn.Linear(5, output_dim)
            ])
        else:
            print('Using FC')
            self.weights = nn.Parameter(torch.rand(input_dim, output_dim), requires_grad=True)
            self.biases = nn.Parameter(torch.rand(input_dim), requires_grad=True)

    def forward(self, x, memory=None):
        if self.dynamic:
            # weights = self.weight_generator(memory).view(x.shape[0], self.num_nodes, self.input_dim, self.output_dim)
            # biases = self.bias_generator(memory).view(x.shape[0], self.num_nodes, self.output_dim)
            weights = self.weight_generator(memory).view(x.shape[0], -1, self.input_dim, self.output_dim)
            biases = self.bias_generator(memory).view(x.shape[0], -1, self.output_dim)
        else:
            weights = self.weights
            biases = self.biases
        return weights, biases
