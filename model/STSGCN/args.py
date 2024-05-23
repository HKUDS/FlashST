import argparse
import numpy as np
import configparser
from lib.predifineGraph import get_adjacency_matrix, load_pickle, weight_matrix
import torch
import pandas as pd

def parse_args(DATASET):
    # get configuration
    config_file = '../conf/STSGCN/{}.conf'.format(DATASET)
    config = configparser.ConfigParser()
    config.read(config_file)
    parser = argparse.ArgumentParser()

    filter_list_str = config.get('model', 'filter_list')
    filter_list = eval(filter_list_str)

    # data
    parser.add_argument('--num_nodes', type=int, default=config['data']['num_nodes'])
    parser.add_argument('--input_window', type=int, default=config['data']['input_window'])
    parser.add_argument('--output_window', type=int, default=config['data']['output_window'])

    # model
    parser.add_argument('--filter_list', type=list, default=config['model']['filter_list'])
    parser.add_argument('--rho', type=int, default=config['model']['rho'])
    parser.add_argument('--feature_dim', type=int, default=config['model']['feature_dim'])
    parser.add_argument('--module_type', type=str, default=config['model']['module_type'])
    parser.add_argument('--activation', type=str, default=config['model']['activation'])
    parser.add_argument('--temporal_emb', type=eval, default=config['model']['temporal_emb'])
    parser.add_argument('--spatial_emb', type=eval, default=config['model']['spatial_emb'])
    parser.add_argument('--use_mask', type=eval, default=config['model']['use_mask'])
    parser.add_argument('--steps', type=int, default=config['model']['steps'])
    parser.add_argument('--first_layer_embedding_size', type=int, default=config['model']['first_layer_embedding_size'])

    args, _ = parser.parse_known_args()
    args.filter_list = filter_list
    args.adj_mx = None
    return args