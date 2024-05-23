import argparse
import numpy as np
import configparser
from lib.predifineGraph import get_adjacency_matrix, load_pickle, weight_matrix
import torch
import pandas as pd

def parse_args(DATASET):
    # get configuration
    config_file = '../conf/TGCN/{}.conf'.format(DATASET)
    config = configparser.ConfigParser()
    config.read(config_file)

    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--num_nodes', type=int, default=config['data']['num_nodes'])
    parser.add_argument('--input_window', type=int, default=config['data']['input_window'])
    parser.add_argument('--output_window', type=int, default=config['data']['output_window'])
    # model
    parser.add_argument('--rnn_units', type=int, default=config['model']['rnn_units'])
    parser.add_argument('--lam', type=float, default=config['model']['lam'])
    parser.add_argument('--output_dim', type=int, default=config['model']['output_dim'])

    args, _ = parser.parse_known_args()
    args.adj_mx = None
    return args