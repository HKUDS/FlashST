import argparse
import numpy as np
import pandas as pd
import configparser
from lib.predifineGraph import get_adjacency_matrix, load_pickle, weight_matrix
import torch

def parse_args(DATASET):
    # get configuration
    config_file = '../conf/MTGNN/{}.conf'.format(DATASET)
    config = configparser.ConfigParser()
    config.read(config_file)

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--device', type=str, default=config['general']['device'])

    # data
    parser.add_argument('--num_nodes', type=int, default=config['data']['num_nodes'])
    parser.add_argument('--input_window', type=int, default=config['data']['input_window'])
    parser.add_argument('--output_window', type=int, default=config['data']['output_window'])
    parser.add_argument('--output_dim', type=int, default=config['data']['output_dim'])
    # model
    parser.add_argument('--gcn_true', type=eval, default=config['model']['gcn_true'])
    parser.add_argument('--buildA_true', type=eval, default=config['model']['buildA_true'])
    parser.add_argument('--gcn_depth', type=int, default=config['model']['gcn_depth'])
    parser.add_argument('--dropout', type=float, default=config['model']['dropout'])
    parser.add_argument('--subgraph_size', type=int, default=config['model']['subgraph_size'])
    parser.add_argument('--node_dim', type=int, default=config['model']['node_dim'])
    parser.add_argument('--dilation_exponential', type=int, default=config['model']['dilation_exponential'])
    parser.add_argument('--conv_channels', type=int, default=config['model']['conv_channels'])
    parser.add_argument('--residual_channels', type=int, default=config['model']['residual_channels'])
    parser.add_argument('--skip_channels', type=int, default=config['model']['skip_channels'])
    parser.add_argument('--end_channels', type=int, default=config['model']['end_channels'])
    parser.add_argument('--layers', type=int, default=config['model']['layers'])
    parser.add_argument('--propalpha', type=float, default=config['model']['propalpha'])
    parser.add_argument('--tanhalpha', type=int, default=config['model']['tanhalpha'])
    parser.add_argument('--layer_norm_affline', type=eval, default=config['model']['layer_norm_affline'])
    parser.add_argument('--use_curriculum_learning', type=eval, default=config['model']['use_curriculum_learning'])
    parser.add_argument('--task_level', type=int, default=config['model']['task_level'])

    args, _ = parser.parse_known_args()
    args.adj_mx = None
    return args