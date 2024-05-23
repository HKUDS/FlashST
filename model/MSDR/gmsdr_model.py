import numpy as np
import torch
import torch.nn as nn

from .gmsdr_cell import GMSDRCell

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2SeqAttrs:
    def __init__(self, args):
        self.max_diffusion_step = args.max_diffusion_step
        self.cl_decay_steps = args.cl_decay_steps
        self.filter_type = args.filter_type
        # self.num_nodes = args.num_nodes
        self.num_rnn_layers = args.num_rnn_layers
        self.rnn_units = args.rnn_units
        # self.hidden_state_size = self.num_nodes * self.rnn_units
        self.pre_k = args.pre_k
        self.pre_v = args.pre_v
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.A_dict = args.A_dict
        self.dataset_use = args.dataset_use
        self.dataset_test = args.dataset_test
        self.mode = args.mode


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, args):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, args)
        self.input_dim = args.input_dim
        self.seq_len = args.seq_len  # for the encoder
        self.mlp = nn.Linear(self.input_dim, self.rnn_units)
        self.gmsdr_layers = nn.ModuleList(
            [GMSDRCell(self.rnn_units, self.input_dim, self.max_diffusion_step, self.pre_k, self.pre_v,
                       self.A_dict, self.dataset_use, self.dataset_test, self.mode,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hx_k, select_dataset):
        """
        Encoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hx_k: (num_layers, batch_size, pre_k, self.num_nodes, self.rnn_units)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hx_k # shape (num_layers, batch_size, pre_k, self.num_nodes, self.rnn_units)
                 (lower indices mean lower layers)
        """
        hx_ks = []
        batch = inputs.shape[0]
        x = inputs.reshape(batch, -1, self.input_dim)
        output = self.mlp(x).view(batch, -1)
        for layer_num, dcgru_layer in enumerate(self.gmsdr_layers):
            next_hidden_state, new_hx_k = dcgru_layer(output, hx_k[layer_num], select_dataset)
            hx_ks.append(new_hx_k)
            output = next_hidden_state
        return output, torch.stack(hx_ks)


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, args):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, args)
        self.output_dim = args.output_dim
        self.horizon = args.horizon  # for the decoder
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.gmsdr_layers = nn.ModuleList(
            [GMSDRCell(self.rnn_units, self.rnn_units, self.max_diffusion_step, self.pre_k, self.pre_v,
                       self.A_dict, self.dataset_use, self.dataset_test, self.mode,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hx_k, select_dataset):
        """
        Decoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.output_dim)
        :param hx_k: (num_layers, batch_size, pre_k, num_nodes, rnn_units)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        hx_ks = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.gmsdr_layers):
            next_hidden_state, new_hx_k = dcgru_layer(output, hx_k[layer_num], select_dataset)
            hx_ks.append(new_hx_k)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        num_nodes = self.A_dict[select_dataset].shape[0]
        output = projected.view(-1, num_nodes * self.output_dim)

        return output, torch.stack(hx_ks)


class GMSDRModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, args):
        super().__init__()
        Seq2SeqAttrs.__init__(self, args)
        self.encoder_model = EncoderModel(args)
        self.decoder_model = DecoderModel(args)
        self.cl_decay_steps = args.cl_decay_steps
        self.use_curriculum_learning = args.use_curriculum_learning
        # self._logger = logger
        self.out = nn.Linear(self.rnn_units, self.decoder_model.output_dim)

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs, select_dataset):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: hx_k: (num_layers, batch_size, pre_k, num_sensor, rnn_units)
        """
        num_nodes = inputs.shape[2]
        hx_k = torch.zeros(self.num_rnn_layers, inputs.shape[1], self.pre_k, num_nodes, self.rnn_units,
                           device=device)
        outputs = []
        for t in range(self.encoder_model.seq_len):
            output, hx_k = self.encoder_model(inputs[t], hx_k, select_dataset)
            outputs.append(output)
        return torch.stack(outputs), hx_k

    def decoder(self, inputs, hx_k, select_dataset, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param inputs: (seq_len, batch_size, num_sensor * rnn_units)
        :param hx_k: (num_layers, batch_size, pre_k, num_sensor, rnn_units)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        decoder_hx_k = hx_k
        decoder_input = inputs

        outputs = []
        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hx_k = self.decoder_model(decoder_input[t],
                                                              decoder_hx_k, select_dataset)
            outputs.append(decoder_output)
        outputs = torch.stack(outputs)
        return outputs


    def forward(self, inputs, select_dataset, labels=None, batches_seen=None):
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        inputs = inputs.transpose(0, 1)
        encoder_outputs, hx_k = self.encoder(inputs, select_dataset)
        # self._logger.debug("Encoder complete, starting decoder")
        outputs = self.decoder(encoder_outputs, hx_k, select_dataset, labels=None, batches_seen=batches_seen)
        # self._logger.debug("Decoder complete")
        # if batches_seen == 0:
        #     self._logger.info(
        #         "Total trainable parameters {}".format(count_parameters(self))
        #     )

        if self.decoder_model.output_dim == 1:
            outputs = outputs.transpose(1, 0).unsqueeze(-1)
        else:
            time_step, batch_size = outputs.shape[0], outputs.shape[1]
            outputs = outputs.transpose(1, 0).unsqueeze(-1).reshape(batch_size, time_step, -1, self.decoder_model.output_dim)
        return outputs
