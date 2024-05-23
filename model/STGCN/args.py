import argparse
import numpy as np
import configparser
from lib.predifineGraph import get_adjacency_matrix, load_pickle, weight_matrix
import torch
import pandas as pd

def scaled_laplacian(W):
    '''
    Normalized graph Laplacian function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :return: np.matrix, [n_route, n_route].
    '''
    # d ->  diagonal degree matrix
    n, d = np.shape(W)[0], np.sum(W, axis=1)
    # L -> graph Laplacian
    L = -W
    L[np.diag_indices_from(L)] = d
    # 返回索引以访问n维数组的主对角线。
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])
    # lambda_max \approx 2.0, the largest eigenvalues of L.L的最大特征值
    # lambda_max = eigs(L, k=1, which='LR')[0][0].real
    lambda_max = np.linalg.eigvals(L).max().real
    return np.mat(2 * L / lambda_max - np.identity(n))


def cheb_poly_approx(L, Ks, n):
    '''
    Chebyshev polynomials approximation function.
    :param L: np.matrix, [n_route, n_route], graph Laplacian.
    :param Ks: int, kernel size of spatial convolution.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, Ks*n_route].
    '''
    L0, L1 = np.mat(np.identity(n)), np.mat(np.copy(L))

    if Ks > 1:
        L_list = [np.copy(L0), np.copy(L1)]
        for i in range(Ks - 2):
            Ln = np.mat(2 * L * L1 - L0)
            L_list.append(np.copy(Ln))
            L0, L1 = np.matrix(np.copy(L1)), np.matrix(np.copy(Ln))
        # L_lsit [Ks, n*n], Lk [n, Ks*n]
        return np.stack(L_list, axis=0)
    elif Ks == 1:
        return np.asarray(L0)
    else:
        raise ValueError(f'ERROR: the size of spatial kernel must be greater than 1, but received "{Ks}".')

def parse_args(DATASET, args_base):
    # get configuration
    config_file = '../conf/STGCN/{}.conf'.format(DATASET)
    config = configparser.ConfigParser()
    config.read(config_file)
    parser = argparse.ArgumentParser(prefix_chars='--', description='predictor_based_arguments')

    blocks1_str = config.get('model', 'blocks1')
    blocks1 = eval(blocks1_str)
    # data
    parser.add_argument('--num_nodes', type=int, default=config['data']['num_nodes'])
    parser.add_argument('--input_window', type=int, default=config['data']['input_window'])
    parser.add_argument('--output_window', type=int, default=config['data']['output_window'])
    # model
    parser.add_argument('--Ks', type=int, default=config['model']['Ks'])
    parser.add_argument('--Kt', type=int, default=config['model']['Kt'])
    parser.add_argument('--blocks1', type=list, default=blocks1)
    parser.add_argument('--drop_prob', type=int, default=config['model']['drop_prob'])
    parser.add_argument('--outputl_ks', type=int, default=config['model']['outputl_ks'])
    args, _ = parser.parse_known_args()

    G_dict = {}
    for data_graph in args_base.dataset_graph:
        L = scaled_laplacian(args_base.A_dict_np[data_graph])
        Lk = cheb_poly_approx(L, 3, args_base.A_dict_np[data_graph].shape[0])
        G_dict[data_graph] = torch.FloatTensor(Lk)

    args.G_dict = G_dict
    return args