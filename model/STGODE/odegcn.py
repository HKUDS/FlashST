import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# Whether use adjoint method or not.
adjoint = False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class ODEFunc(nn.Module):

    def __init__(self, feature_dim, temporal_dim, adj, dataset_use, dataset_test, mode, device):
        super(ODEFunc, self).__init__()
        self.adj = adj
        self.x0 = None
        self.mode = mode
        self.dataset2index = {}
        if mode == 'pretrain':
            self.alpha_pretrain = []
            for i, data_graph in enumerate(dataset_use):
                self.dataset2index[data_graph] = i
                n_dataset = adj[data_graph].shape[1]
                self.alpha_pretrain.append(nn.Parameter(0.8 * torch.ones(n_dataset)).to(device))
        else:
            self.alpha_eval = []
            for i, data_graph in enumerate([dataset_test]):
                self.dataset2index[data_graph] = i
                n_dataset = adj[data_graph].shape[1]
                self.alpha_eval.append(nn.Parameter(0.8 * torch.ones(n_dataset)).to(device))

        # self.alpha = nn.Parameter(0.8 * torch.ones(adj.shape[1]))
        self.beta = 0.6
        self.w = nn.Parameter(torch.eye(feature_dim))
        self.d = nn.Parameter(torch.zeros(feature_dim) + 1)
        self.w2 = nn.Parameter(torch.eye(temporal_dim))
        self.d2 = nn.Parameter(torch.zeros(temporal_dim) + 1)

    def forward(self, t, x, select_dataset):
        if self.mode == 'pretrain':
            alpha = torch.sigmoid(self.alpha_pretrain[self.dataset2index[select_dataset]]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        else:
            alpha = torch.sigmoid(self.alpha_eval[self.dataset2index[select_dataset]]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

        # alpha = torch.sigmoid(self.alpha).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        # print(self.adj.shape, x.shape)
        xa = torch.einsum('ij, kjlm->kilm', self.adj[select_dataset], x)

        # ensure the eigenvalues to be less than 1
        d = torch.clamp(self.d, min=0, max=1)
        w = torch.mm(self.w * d, torch.t(self.w))
        xw = torch.einsum('ijkl, lm->ijkm', x, w)

        d2 = torch.clamp(self.d2, min=0, max=1)
        w2 = torch.mm(self.w2 * d2, torch.t(self.w2))
        xw2 = torch.einsum('ijkl, km->ijml', x, w2)

        f = alpha / 2 * xa - x + xw - x + xw2 - x + self.x0
        return f


class ODEblock(nn.Module):
    def __init__(self, odefunc, t=torch.tensor([0,1])):
        super(ODEblock, self).__init__()
        self.t = t
        self.odefunc = odefunc

    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()



    def forward(self, x, select_dataset):
        def wrapped_odefunc(t, x):
            return self.odefunc(t, x, select_dataset)
        t = self.t.type_as(x)
        z = odeint(wrapped_odefunc, x, t, method='euler')[1]
        # z = odeint(self.odefunc, x, t, method='euler')[1]
        return z


# Define the ODEGCN model.
class ODEG(nn.Module):
    def __init__(self, feature_dim, temporal_dim, adj, time, dataset_use, dataset_test, mode, device):
        super(ODEG, self).__init__()
        self.odeblock = ODEblock(ODEFunc(feature_dim, temporal_dim, adj, dataset_use, dataset_test, mode, device), t=torch.tensor([0, time]))

    def forward(self, x, select_dataset=None):
        self.odeblock.set_x0(x)
        z = self.odeblock(x, select_dataset)
        return F.relu(z)
