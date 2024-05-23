import argparse
import numpy as np
import configparser
import torch
from scipy.sparse.linalg import eigs
from lib.predifineGraph import load_pickle, weight_matrix, get_adjacency_matrix
import pandas as pd

# def get_adjacency_matrix(distance_df_filename, num_of_vertices,
#                          type_='connectivity', id_filename=None):
#     '''
#     Parameters
#     ----------
#     distance_df_filename: str, path of the csv file contains edges information
#     num_of_vertices: int, the number of vertices
#     type_: str, {connectivity, distance}
#     Returns
#     ----------
#     A: np.ndarray, adjacency matrix
#     '''
#     import csv
#
#     A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
#                  dtype=np.float32)
#
#     if id_filename != 'None':
#         with open(id_filename, 'r') as f:
#             id_dict = {int(i): idx
#                        for idx, i in enumerate(f.read().strip().split('\n'))}
#         with open(distance_df_filename, 'r') as f:
#             f.readline()
#             reader = csv.reader(f)
#             for row in reader:
#                 if len(row) != 3:
#                     continue
#                 i, j, distance = int(row[0]), int(row[1]), float(row[2])
#                 A[id_dict[i], id_dict[j]] = 1
#                 A[id_dict[j], id_dict[i]] = 1
#         return A
#
#     # Fills cells in the matrix with distances.
#     with open(distance_df_filename, 'r') as f:
#         f.readline()
#         reader = csv.reader(f)
#         for row in reader:
#             if len(row) != 3:
#                 continue
#             i, j, distance = int(row[0]), int(row[1]), float(row[2])
#             if type_ == 'connectivity':
#                 A[i, j] = 1
#                 A[j, i] = 1
#             elif type_ == 'distance':
#                 A[i, j] = 1 / distance
#                 A[j, i] = 1 / distance
#             else:
#                 raise ValueError("type_ error, must be "
#                                  "connectivity or distance!")
#     return A



def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])

def parse_args(DATASET):
    # get configuration
    config_file = '../conf/ST-WA/{}.conf'.format(DATASET)
    config = configparser.ConfigParser()
    config.read(config_file)

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=config['general']['device'], type=str)
    parser.add_argument('--data', default=DATASET, help='data path', type=str, )
    # parser.add_argument('--adj_filename', type=str, default=config['data']['adj_filename'])
    parser.add_argument('--id_filename', type=str, default=config['data']['id_filename'])
    parser.add_argument('--val_ratio', type=float, default=config['data']['val_ratio'])
    parser.add_argument('--test_ratio', type=float, default=config['data']['test_ratio'])
    parser.add_argument('--num_nodes', type=int, default=config['data']['num_nodes'])
    parser.add_argument('--lag', type=int, default=config['data']['lag'])
    parser.add_argument('--horizon', type=int, default=config['data']['horizon'])

    parser.add_argument('--in_dim', type=int, default=config['model']['in_dim'])
    parser.add_argument('--out_dim', type=int, default=config['model']['out_dim'])
    parser.add_argument('--channels', type=int, default=config['model']['channels'])
    parser.add_argument('--dynamic', type=str, default=config['model']['dynamic'])
    parser.add_argument('--memory_size', type=int, default=config['model']['memory_size'])

    parser.add_argument('--column_wise', type=bool, default=False)

    args, _ = parser.parse_known_args()

    args.supports = None
    return args