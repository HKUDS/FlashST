import argparse
import numpy as np
import os
from fastdtw import fastdtw
from tqdm import tqdm
import csv
import torch
import pandas as pd

def read_data(args):
    """read data, generate spatial adjacency matrix and semantic adjacency matrix by dtw

    Args:
        sigma1: float, default=0.1, sigma for the semantic matrix
        sigma2: float, default=10, sigma for the spatial matrix
        thres1: float, default=0.6, the threshold for the semantic matrix
        thres2: float, default=0.5, the threshold for the spatial matrix

    Returns:
        data: tensor, T * N * 1
        dtw_matrix: array, semantic adjacency matrix
        sp_matrix: array, spatial adjacency matrix
    """
    filename = args.filename
    # filepath = "./data/"
    filepath = args.filepath
    # if args.remote:
    #     filepath = '/home/lantu.lqq/ftemp/data/'
    data = np.load(filepath + filename + '.npz')['data']
    if data.ndim != 3:
        data = np.expand_dims(data, axis=-1)
    # print(data.shape)
    # PEMS04 == shape: (16992, 307, 3)    feature: flow,occupy,speed
    # PEMSD7M == shape: (12672, 228, 1)
    # PEMSD7L == shape: (12672, 1026, 1)
    num_node = data.shape[1]
    mean_value = np.mean(data, axis=(0, 1)).reshape(1, 1, -1)
    std_value = np.std(data, axis=(0, 1)).reshape(1, 1, -1)
    data = (data - mean_value) / std_value
    mean_value = mean_value.reshape(-1)[0]
    std_value = std_value.reshape(-1)[0]
    # print(sss)

    if not os.path.exists(f'../data/STGODE/{filename}/{filename}_dtw_distance.npy'):
        data_mean = np.mean([data[:, :, 0][24 * 12 * i: 24 * 12 * (i + 1)] for i in range(data.shape[0] // (24 * 12))],
                            axis=0)
        data_mean = data_mean.squeeze().T
        dtw_distance = np.zeros((num_node, num_node))
        for i in tqdm(range(num_node)):
            for j in range(i, num_node):
                dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
        for i in range(num_node):
            for j in range(i):
                dtw_distance[i][j] = dtw_distance[j][i]
        np.save(f'../data/STGODE/{filename}/{filename}_dtw_distance.npy', dtw_distance)

    dist_matrix = np.load(f'../data/STGODE/{filename}/{filename}_dtw_distance.npy')

    mean = np.mean(dist_matrix)
    std = np.std(dist_matrix)
    dist_matrix = (dist_matrix - mean) / std
    sigma = args.sigma1
    dist_matrix = np.exp(-dist_matrix ** 2 / sigma ** 2)
    dtw_matrix = np.zeros_like(dist_matrix)
    dtw_matrix[dist_matrix > args.thres1] = 1

    # # use continuous semantic matrix
    # if not os.path.exists(f'data/{filename}_dtw_c_matrix.npy'):
    #     dist_matrix = np.load(f'data/{filename}_dtw_distance.npy')
    #     # normalization
    #     std = np.std(dist_matrix[dist_matrix != np.float('inf')])
    #     mean = np.mean(dist_matrix[dist_matrix != np.float('inf')])
    #     dist_matrix = (dist_matrix - mean) / std
    #     sigma = 0.1
    #     dtw_matrix = np.exp(- dist_matrix**2 / sigma**2)
    #     dtw_matrix[dtw_matrix < 0.5] = 0
    #     np.save(f'data/{filename}_dtw_c_matrix.npy', dtw_matrix)
    # dtw_matrix = np.load(f'data/{filename}_dtw_c_matrix.npy')

    # use continuous spatial matrix
    if not os.path.exists(f'../data/STGODE/{filename}/{filename}_spatial_distance.npy'):
        if filename == 'PEMS07M':
            dist_matrix = pd.read_csv(filepath + filename + '.csv', header=None).values
        elif filename == 'NYC_BIKE':
            dist_matrix = pd.read_csv(filepath + filename + '.csv', header=None).values.astype(np.float32)
            dist_matrix = (1 - dist_matrix) * 1000
        elif filename == 'chengdu_didi':
            dist_matrix = np.load(filepath + 'matrix.npy').astype(np.float32)
            dist_matrix = (1 - dist_matrix) * 1000
        elif filename == 'CA_District5':
            dist_matrix = np.load(filepath + filename + '.npy').astype(np.float32)
            dist_matrix = (1 - dist_matrix) * 1000
        elif filename == 'PEMS03':
            with open(filepath + filename + '.txt', 'r') as f:
                id_dict = {int(i): idx for idx, i in
                           enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    dist_matrix[id_dict[i], id_dict[j]] = distance
        else:
            dist_matrix = np.zeros((num_node, num_node)) + np.float('inf')
            with open(filepath + filename + '.csv', 'r') as fp:
                file = csv.reader(fp)
            for line in file:
                break
            for line in file:
                print(line)
                print(line[0])
                start = int(line[0])
                end = int(line[1])
                dist_matrix[start][end] = float(line[2])
                dist_matrix[end][start] = float(line[2])
        np.save(f'../data/STGODE/{filename}/{filename}_spatial_distance.npy', dist_matrix)

    # use 0/1 spatial matrix
    # if not os.path.exists(f'data/{filename}_sp_matrix.npy'):
    #     dist_matrix = np.load(f'data/{filename}_spatial_distance.npy')
    #     sp_matrix = np.zeros((num_node, num_node))
    #     sp_matrix[dist_matrix != np.float('inf')] = 1
    #     np.save(f'data/{filename}_sp_matrix.npy', sp_matrix)
    # sp_matrix = np.load(f'data/{filename}_sp_matrix.npy')

    dist_matrix = np.load(f'../data/STGODE/{filename}/{filename}_spatial_distance.npy')
    # normalization
    std = np.std(dist_matrix[dist_matrix != np.float('inf')])
    mean = np.mean(dist_matrix[dist_matrix != np.float('inf')])
    dist_matrix = (dist_matrix - mean) / std
    sigma = args.sigma2
    sp_matrix = np.exp(- dist_matrix ** 2 / sigma ** 2)
    sp_matrix[sp_matrix < args.thres2] = 0
    # np.save(f'data/{filename}_sp_c_matrix.npy', sp_matrix)
    # sp_matrix = np.load(f'data/{filename}_sp_c_matrix.npy')

    print(f'average degree of spatial graph is {np.sum(sp_matrix > 0) / 2 / num_node}')
    print(f'average degree of semantic graph is {np.sum(dtw_matrix > 0) / 2 / num_node}')
    return torch.from_numpy(data.astype(np.float32)), mean_value, std_value, dtw_matrix, sp_matrix

def get_normalized_adj(A):
    """
    Returns a tensor, the degree normalized adjacency matrix.
    """
    alpha = 0.8
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    A_reg = alpha / 2 * (np.eye(A.shape[0]) + A_wave)
    return torch.from_numpy(A_reg.astype(np.float32))


def parse_args(DATASET, args_base):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--num-gpu', type=int, default=0, help='the number of the gpu to use')
    # parser.add_argument('--batch-size', type=int, default=16, help='batch size')

    # parser.add_argument('--filename', type=str, default='pems08')
    # parser.add_argument('--filepath', type=str, default='F:/data/traffic_data/PEMS_data/')
    # parser.add_argument('--train-ratio', type=float, default=0.6, help='the ratio of training dataset')
    # parser.add_argument('--valid-ratio', type=float, default=0.2, help='the ratio of validating dataset')
    # parser.add_argument('--his-length', type=int, default=12, help='the length of history time series of input')
    # parser.add_argument('--pred-length', type=int, default=12, help='the length of target time series for prediction')

    parser.add_argument('--sigma1', type=float, default=0.1, help='sigma for the semantic matrix')
    parser.add_argument('--sigma2', type=float, default=10, help='sigma for the spatial matrix')
    parser.add_argument('--thres1', type=float, default=0.6, help='the threshold for the semantic matrix')
    parser.add_argument('--thres2', type=float, default=0.5, help='the threshold for the spatial matrix')
    # parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')

    args, _ = parser.parse_known_args()

    A_sp_wave_dict = {}
    A_se_wave_dict = {}

    for data_graph in args_base.dataset_graph:
        args.filepath = '../data/' + data_graph +'/'
        args.filename = data_graph
        _, _, _, dtw_matrix, sp_matrix = read_data(args)
        A_sp_wave_dict[data_graph], A_se_wave_dict[data_graph] = get_normalized_adj(sp_matrix).to(args_base.device), get_normalized_adj(dtw_matrix).to(args_base.device)
        # args.A_sp_wave, args.A_se_wave = get_normalized_adj(sp_matrix), get_normalized_adj(dtw_matrix)
    args.A_sp_wave_dict, args.A_se_wave_dict = A_sp_wave_dict, A_se_wave_dict
    return args