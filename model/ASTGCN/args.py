import argparse
import numpy as np
import configparser
from lib.predifineGraph import get_adjacency_matrix, load_pickle, weight_matrix
import torch
import pandas as pd

def parse_args(DATASET, args_base):
    # get configuration
    config_file = '../conf/ASTGCN/{}.conf'.format(DATASET)
    config = configparser.ConfigParser()
    config.read(config_file)

    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--num_nodes', type=int, default=config['data']['num_nodes'])
    parser.add_argument('--len_input', type=int, default=config['data']['len_input'])
    parser.add_argument('--num_for_predict', type=int, default=config['data']['num_for_predict'])
    # model
    parser.add_argument('--nb_block', type=int, default=config['model']['nb_block'])
    parser.add_argument('--K', type=int, default=config['model']['K'])
    parser.add_argument('--nb_chev_filter', type=int, default=config['model']['nb_chev_filter'])
    parser.add_argument('--nb_time_filter', type=int, default=config['model']['nb_time_filter'])
    parser.add_argument('--time_strides', type=int, default=config['model']['time_strides'])
    args, _ = parser.parse_known_args()
    return args