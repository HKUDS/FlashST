import torch
import torch.nn as nn
from model.AGCRN.AGCRNCell import AGCRNCell

class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.out_dim = dim_out
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        # assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size, node_dataset):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size, node_dataset))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

class AGCRN(nn.Module):
    def __init__(self, args, dim_in, dim_out, A_dict, dataset_use, data_test, mode):
        super(AGCRN, self).__init__()
        self.A_dict = A_dict
        self.mode = mode
        self.num_node = args.num_nodes
        self.input_dim = dim_in
        self.hidden_dim = args.rnn_units
        self.output_dim = dim_out
        self.horizon = args.horizon
        self.num_layers = args.num_layers

        self.default_graph = args.default_graph

        self.dataset2index = {}
        if mode == 'pretrain':
            self.neb_pretrain = []
            for i, data_graph in enumerate(dataset_use):
                self.dataset2index[data_graph] = i
                n_dataset = A_dict[data_graph].shape[0]
                self.neb_pretrain.append(nn.Parameter(torch.randn(n_dataset, args.embed_dim).to(args.device), requires_grad=True))
        else:
            self.neb_eval = []
            for i, data_graph in enumerate([data_test]):
                self.dataset2index[data_graph] = i
                n_dataset = A_dict[data_graph].shape[0]
                self.neb_eval.append(nn.Parameter(torch.randn(n_dataset, args.embed_dim).to(args.device), requires_grad=True))


        # self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)

        self.Lin_input = nn.Linear(self.hidden_dim, 1)

        self.encoder = AVWDCRNN(args.num_nodes, self.input_dim, args.rnn_units, args.cheb_k,
                                args.embed_dim, args.num_layers)

        #predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, source, select_dataset):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)


        init_state = self.encoder.init_hidden(source.shape[0], self.A_dict[select_dataset].shape[0])
        # source = self.Lin_input(source)
        if self.mode == 'pretrain':
            output, _ = self.encoder(source, init_state, self.neb_pretrain[self.dataset2index[select_dataset]])      #B, T, N, hidden
        else:
            output, _ = self.encoder(source, init_state, self.neb_eval[self.dataset2index[select_dataset]])  # B, T, N, hidden
        output = output[:, -1:, :, :]                                   #B, 1, N, hidden

        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.A_dict[select_dataset].shape[0])
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C

        return output