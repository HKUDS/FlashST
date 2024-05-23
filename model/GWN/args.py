import argparse
import numpy as np
import pandas as pd
import configparser
from lib.predifineGraph import get_adjacency_matrix, load_pickle, weight_matrix
import torch

def parse_args(DATASET):
    # get configuration
    config_file = '../conf/GWN/{}.conf'.format(DATASET)
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
    parser.add_argument('--dropout', type=float, default=config['model']['dropout'])
    parser.add_argument('--blocks', type=int, default=config['model']['blocks'])
    parser.add_argument('--layers', type=int, default=config['model']['layers'])
    parser.add_argument('--gcn_bool', type=eval, default=config['model']['gcn_bool'])
    parser.add_argument('--addaptadj', type=eval, default=config['model']['addaptadj'])
    parser.add_argument('--adjtype', type=str, default=config['model']['adjtype'])
    parser.add_argument('--randomadj', type=eval, default=config['model']['randomadj'])
    parser.add_argument('--aptonly', type=eval, default=config['model']['aptonly'])
    parser.add_argument('--kernel_size', type=int, default=config['model']['kernel_size'])
    parser.add_argument('--nhid', type=int, default=config['model']['nhid'])
    parser.add_argument('--residual_channels', type=int, default=config['model']['residual_channels'])
    parser.add_argument('--dilation_channels', type=int, default=config['model']['dilation_channels'])

    args, _ = parser.parse_known_args()
    args.adj_mx = None
    return args