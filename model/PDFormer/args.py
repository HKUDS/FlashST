import os
import numpy as np
from fastdtw import fastdtw
from tqdm import tqdm
from tslearn.clustering import TimeSeriesKMeans, KShape
from lib.data_process import load_st_dataset
import argparse
import configparser

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def Add_Window_Horizon(data, window=3, horizon=1, single=False):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []      #windows
    Y = []      #horizon
    index = 0
    if single:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window+horizon-1:index+window+horizon])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window:index+window+horizon])
            index = index + 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def get_dtw(df, args, dataset, num_nodes):
    filename = dataset
    cache_path = f'../data/PDFormer/{filename}/{filename}_dtw.npy'
    if not os.path.exists(cache_path):
        data_mean = np.mean(
            [df[24 * args.points_per_hour * i: 24 * args.points_per_hour * (i + 1)]
             for i in range(df.shape[0] // (24 * args.points_per_hour))], axis=0)
        dtw_distance = np.zeros((num_nodes, num_nodes))
        for i in tqdm(range(num_nodes)):
            for j in range(i, num_nodes):
                dtw_distance[i][j], _ = fastdtw(data_mean[:, i, :], data_mean[:, j, :], radius=6)
        for i in range(num_nodes):
            for j in range(i):
                dtw_distance[i][j] = dtw_distance[j][i]
        np.save(cache_path, dtw_distance)
    dtw_matrix = np.load(cache_path)
    print('Load DTW matrix from {}'.format(cache_path))
    return dtw_matrix

def get_pattern_key(x_train, args, dataset, args_base):
    filename = dataset
    cache_path = f'../data/PDFormer/{filename}/{filename}_pattern.npy'
    if not os.path.exists(cache_path):
        cand_key_time_steps = args.cand_key_days * args.points_per_day
        pattern_cand_keys = x_train[:cand_key_time_steps, :args.s_attn_size, :, :args_base.output_dim].swapaxes(1,2).reshape(
            -1, args.s_attn_size, args_base.output_dim)
        print("Clustering...")
        if args.cluster_method == "kshape":
            km = KShape(n_clusters=args.n_cluster, max_iter=args.cluster_max_iter).fit(pattern_cand_keys)
        else:
            km = TimeSeriesKMeans(n_clusters=args.n_cluster, metric="softdtw", max_iter=args.cluster_max_iter).fit(
                pattern_cand_keys)
        pattern_keys = km.cluster_centers_
        np.save(cache_path, pattern_keys)
        print("Saved at file " + cache_path)
    pattern_key_matrix = np.load(cache_path)
    return pattern_key_matrix

def load_rel(adj_mx, args, dataset):
    filename = dataset
    cache_path = f'../data/PDFormer/{filename}/{filename}_sh_mx.npy'
    sh_mx = adj_mx.copy()
    if args.type_short_path == 'hop':
        if not os.path.exists(cache_path):
            print('Max adj_mx value = {}'.format(adj_mx.max()))
            num_nodes = adj_mx.shape[0]
            sh_mx[sh_mx > 0] = 1
            sh_mx[sh_mx == 0] = 511
            for i in range(num_nodes):
                sh_mx[i, i] = 0
            for k in range(num_nodes):
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        sh_mx[i, j] = min(sh_mx[i, j], sh_mx[i, k] + sh_mx[k, j], 511)
            np.save(cache_path, sh_mx)
        sh_mx = np.load(cache_path)
    return sh_mx


def parse_args(DATASET, args_base):
    # get configuration
    print(DATASET)
    config_file = '../conf/PDFormer/{}.conf'.format(DATASET)
    config = configparser.ConfigParser()
    config.read(config_file)

    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--embed_dim', type=int, default=config['model']['embed_dim'])
    parser.add_argument('--skip_dim', type=int, default=config['model']['skip_dim'])
    parser.add_argument('--lape_dim', type=int, default=config['model']['lape_dim'])

    parser.add_argument('--geo_num_heads', type=int, default=config['model']['geo_num_heads'])
    parser.add_argument('--sem_num_heads', type=int, default=config['model']['sem_num_heads'])
    parser.add_argument('--t_num_heads', type=int, default=config['model']['t_num_heads'])
    parser.add_argument('--mlp_ratio', type=int, default=config['model']['mlp_ratio'])
    parser.add_argument('--qkv_bias', type=eval, default=config['model']['qkv_bias'])
    parser.add_argument('--drop', type=float, default=config['model']['drop'])
    parser.add_argument('--attn_drop', type=float, default=config['model']['attn_drop'])
    parser.add_argument('--drop_path', type=float, default=config['model']['drop_path'])
    parser.add_argument('--s_attn_size', type=int, default=config['model']['s_attn_size'])
    parser.add_argument('--t_attn_size', type=int, default=config['model']['t_attn_size'])
    parser.add_argument('--enc_depth', type=int, default=config['model']['enc_depth'])
    parser.add_argument('--type_ln', type=str, default=config['model']['type_ln'])
    parser.add_argument('--type_short_path', type=str, default=config['model']['type_short_path'])
    parser.add_argument('--add_time_in_day', type=eval, default=config['model']['add_time_in_day'])
    parser.add_argument('--add_day_in_week', type=eval, default=config['model']['add_day_in_week'])

    parser.add_argument('--far_mask_delta', type=int, default=config['model']['far_mask_delta'])
    parser.add_argument('--dtw_delta', type=int, default=config['model']['dtw_delta'])
    parser.add_argument('--time_intervals', type=int, default=config['model']['time_intervals'])
    parser.add_argument('--cand_key_days', type=int, default=config['model']['cand_key_days'])
    parser.add_argument('--n_cluster', type=int, default=config['model']['n_cluster'])
    parser.add_argument('--cluster_max_iter', type=int, default=config['model']['cluster_max_iter'])
    parser.add_argument('--cluster_method', type=str, default=config['model']['cluster_method'])

    # self.s_attn_size = config.get("s_attn_size", 3)
    # self.n_cluster = config.get("n_cluster", 16)
    # self.cluster_max_iter = config.get("cluster_max_iter", 5)
    # self.cluster_method = config.get("cluster_method", "kshape")

    args_predictor, _ = parser.parse_known_args()

    args_predictor.points_per_hour = 3600 // args_predictor.time_intervals
    args_predictor.points_per_day = 24 * 3600 // args_predictor.time_intervals

    if args_base.mode == 'pretrain':
        data_list = args_base.dataset_use
    else:
        data_list = [args_base.dataset_test]
    dtw_matrix_dict = {}
    sh_mx_dict = {}
    pattern_key_matrix_dict = {}
    adj_mx_dict = {}

    for i, data_graph in enumerate(data_list):
        data = load_st_dataset(data_graph, args_base)
        data = data[..., 0:args_base.input_base_dim]
        data_train, data_val, data_test = split_data_by_ratio(data, args_base.val_ratio, args_base.test_ratio)
        x_tra, y_tra = Add_Window_Horizon(data_train, args_base.his, args_base.pred)
        num_nodes = args_base.A_dict_np[data_graph].shape[0]

        dtw_matrix = get_dtw(data, args_predictor, data_graph, num_nodes)
        dtw_matrix_dict[data_graph] = dtw_matrix
        sh_mx = load_rel(args_base.A_dict_np[data_graph], args_predictor, data_graph)
        sh_mx_dict[data_graph] = sh_mx
        pattern_key_matrix = get_pattern_key(x_tra, args_predictor, data_graph, args_base)
        pattern_key_matrix_dict[data_graph] = pattern_key_matrix
    args_predictor.dtw_matrix_dict = dtw_matrix_dict
    args_predictor.adj_mx_dict = args_base.A_dict_np
    args_predictor.sd_mx = None
    args_predictor.sh_mx_dict = sh_mx_dict
    args_predictor.pattern_key_matrix_dict = pattern_key_matrix_dict
    args_predictor.lap_mx_dict = args_base.lpls_dict
    # print(sss)
    return args_predictor