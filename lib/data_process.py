import torch
import numpy as np
import random
import os
from torch.utils.data import Dataset

def time_add(data, week_start, interval=5, weekday_only=False, holiday_list=None, day_start=0, hour_of_day=24):
    # day and week
    if weekday_only:
        week_max = 5
    else:
        week_max = 7
    time_slot = hour_of_day * 60 // interval
    day_data = np.zeros_like(data)
    week_data = np.zeros_like(data)
    holiday_data = np.zeros_like(data)
    day_init = day_start
    week_init = week_start
    holiday_init = 1
    for index in range(data.shape[0]):
        if (index) % time_slot == 0:
            day_init = day_start
        day_init = day_init + 1 * (interval // 5)
        if (index) % time_slot == 0 and index !=0:
            week_init = week_init + 1
        if week_init > week_max:
            week_init = 1
        if day_init < 6:
            holiday_init = 1
        else:
            holiday_init = 2

        day_data[index:index + 1, :] = day_init
        week_data[index:index + 1, :] = week_init
        holiday_data[index:index + 1, :] = holiday_init

    if holiday_list is None:
        k = 1
    else:
        for j in holiday_list :
            holiday_data[j-1 * time_slot:j * time_slot, :] = 2
    return day_data, week_data, holiday_data

def data_type_init(DATASET, args):
    if DATASET == 'METR_LA' or DATASET == 'SZ_TAXI' or DATASET == 'PEMS07M':
        data_type = 'speed'
    elif DATASET == 'PEMS08' or DATASET == 'PEMS04' or DATASET == 'PEMS03' or DATASET == 'PEMS07':
        data_type = 'flow'
    elif DATASET == 'NYC_BIKE' or DATASET == 'NYC_TAXI' or DATASET == 'CHI_TAXI' or DATASET == 'CHI_BIKE':
        data_type = 'demand'
    elif DATASET == 'Electricity':
        data_type = 'MTS'
    elif DATASET == 'NYC_CRIME' or DATASET == 'CHI_CRIME':
        data_type = 'crime'
    elif DATASET == 'BEIJING_SUBWAY':
        data_type = 'people flow'
    elif DATASET == 'chengdu_didi' or 'shenzhen_didi':
        data_type = 'index'
    else:
        raise ValueError

    args.data_type = data_type

# load dataset
def load_st_dataset(dataset, args):
    #output B, N, D
    # 1 / 1 / 2018 - 2 / 28 / 2018 Monday
    if dataset == 'PEMS04':
        data_path = os.path.join('../data/PEMS04/PEMS04.npz')
        data = np.load(data_path)['data'][:, :, 0]  # onley the first dimension, traffic flow data
        print(data.shape, data[data==0].shape)
        week_start = 1
        interval = 5
        week_day = 7
        args.interval = interval
        args.week_day = week_day
        holiday_list = [1, 15, 50]
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)

    # 7 / 1 / 2016 - 8 / 31 / 2016 Friday
    elif dataset == 'PEMS08':
        data_path = os.path.join('../data/PEMS08/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, 0]  # only the first dimension, traffic flow data
        print(data.shape, data[data==0].shape)
        week_start = 5
        holiday_list = [4]
        interval = 5
        week_day = 7
        args.interval = interval
        args.week_day = week_day
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)

    #   9/1/2018 - 11/30/2018 Saturday
    elif dataset == 'PEMS03':
        data_path = os.path.join('../data/PEMS03/PEMS03.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
        week_start = 6
        interval = 5
        week_day = 7
        holiday_list = None
        args.interval = interval
        args.week_day = week_day
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)

    # 5 / 1 / 2017 - 8 / 31 / 2017 Monday
    elif dataset == 'PEMS07':
        data_path = os.path.join('../data/PEMS07/PEMS07.npz')
        data = np.load(data_path)['data'][:, :, 0]  # only the first dimension, traffic flow data
        week_start = 1
        interval = 5
        week_day = 7
        holiday_list = None
        args.interval = interval
        args.week_day = week_day
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)

    # 1 / 1 / 2017 - 2 / 28 / 2017 Sunday
    elif dataset == 'CA_District5':
        data_path = os.path.join('../data/CA_District5/CA_District5.npz')
        data = np.load(data_path)['data'][:, :]  # only the first dimension, traffic flow data
        week_start = 7
        interval = 5
        week_day = 7
        holiday_list = None
        args.interval = interval
        args.week_day = week_day
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)

    # 1 / 1 / 2018 - 4 / 30 / 2018 Monday
    elif dataset == 'chengdu_didi':
        data_path = os.path.join('../data/chengdu_didi/chengdu_didi.npz')
        data = np.load(data_path)['data'][:, :, 0]  # only the first dimension, traffic index
        print(data.shape, data[data==0].shape)

        week_start = 1
        holiday_list = [4]
        interval = 10
        week_day = 7
        args.interval = interval
        args.week_day = week_day
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)

    # 5 / 1 / 2012 - 6 / 30 / 2012 Tuesday
    elif dataset == 'PEMS07M':
        data_path = os.path.join('../data/PEMS07M/PEMS07M.npz')
        data = np.load(data_path)['data']  # only traffic speed data
        week_start = 2
        weekday_only = True
        interval = 5
        week_day = 5
        args.interval = interval
        args.week_day = week_day
        holiday_list = []
        day_data, week_data, holiday_data = time_add(data, week_start, interval, weekday_only, holiday_list=holiday_list)

    elif dataset == 'NYC_BIKE':
        data_path = os.path.join('../data/NYC_BIKE/NYC_BIKE.npz')
        data = np.load(data_path)['data'][..., 0].astype(np.float64)
        print(data.dtype,)
        week_start = 5
        weekday_only = False
        interval = 30
        week_day = 7
        args.interval = interval
        args.week_day = week_day
        holiday_list = []
        day_data, week_data, holiday_data = time_add(data, week_start, interval, weekday_only,
                                                     holiday_list=holiday_list)
    else:
        raise ValueError

    args.num_nodes = data.shape[1]

    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
        day_data = np.expand_dims(day_data, axis=-1).astype(int)
        week_data = np.expand_dims(week_data, axis=-1).astype(int)
        # holiday_data = np.expand_dims(holiday_data, axis=-1).astype(int)
        data = np.concatenate([data, day_data, week_data], axis=-1)
    elif len(data.shape) > 2:
        day_data = np.expand_dims(day_data, axis=-1).astype(int)
        week_data = np.expand_dims(week_data, axis=-1).astype(int)
        data = np.concatenate([data, day_data, week_data], axis=-1)
    else:
        raise ValueError

    print('Load %s Dataset shaped: ' % dataset, data.shape, data[..., 0:1].max(), data[..., 0:1].min(),
          data[..., 0:1].mean(), np.median(data[..., 0:1]), data.dtype)
    return data

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

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean

def normalize_dataset(data, data_type, input_base_dim, normalize_type="meanstd"):
    if normalize_type == 'maxmin':
        data_ori = data[:, :, 0:input_base_dim]
        data_day = data[:, :, input_base_dim:input_base_dim+1]
        data_week = data[:, :, input_base_dim+1:input_base_dim+2]

        max_data = data_ori.max()
        min_data = data_ori.min()
        mean_day = data_day.mean()
        std_week = data_week.std()

        std_day = data_day.std()
        mean_week = data_week.mean()
        scaler_data = StandardScaler(min_data, max_data-min_data)
        scaler_day = StandardScaler(mean_day, std_day)
        scaler_week = StandardScaler(mean_week, std_week)
    else:
        data_ori = data[:, :, 0:input_base_dim]
        data_day = data[:, :, input_base_dim:input_base_dim+1]
        data_week = data[:, :, input_base_dim+1:input_base_dim+2]
        # data_holiday = data[:, :, 3:4]

        mean_data = data_ori.mean()
        std_data = data_ori.std()
        mean_day = data_day.mean()
        std_day = data_day.std()
        mean_week = data_week.mean()
        std_week = data_week.std()

        scaler_data = StandardScaler(mean_data, std_data)
        scaler_day = StandardScaler(mean_day, std_day)
        scaler_week = StandardScaler(mean_week, std_week)
    print('Normalize the dataset by Standard Normalization')
    return scaler_data, scaler_day, scaler_week, None

def define_dataloder(stage, args):
    x_trn_dict, y_trn_dict = {}, {}
    x_val_dict, y_val_dict = {}, {}
    x_tst_dict, y_tst_dict = {}, {}
    scaler_dict = {}

    datause_keys = args.dataset_use
    datatst_keys = args.dataset_test

    data_inlist = []
    if stage == 'eval':
        data_inlist.append(datatst_keys)
    else:
        data_inlist = datause_keys

    for dataset_name in data_inlist:
        print(data_inlist, dataset_name, args.val_ratio, args.test_ratio)
        # print(sss)
        data = load_st_dataset(dataset_name, args)
        data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)

        scaler_data, scaler_day, scaler_week, scaler_holiday = normalize_dataset(data_train, args.data_type, args.input_base_dim)
        print(data_train.shape, scaler_data.mean, scaler_data.std)
        x_tra, y_tra = Add_Window_Horizon(data_train, args.his, args.pred)
        x_val, y_val = Add_Window_Horizon(data_val, args.his, args.pred)
        x_test, y_test = Add_Window_Horizon(data_test, args.his, args.pred)

        if args.real_value == False:
            x_tra[..., :args.input_base_dim] = scaler_data.transform(x_tra[:, :, :, :args.input_base_dim])
            y_tra[..., :args.input_base_dim] = scaler_data.transform(y_tra[:, :, :, :args.input_base_dim])
            x_val[..., :args.input_base_dim] = scaler_data.transform(x_val[:, :, :, :args.input_base_dim])
            y_val[..., :args.input_base_dim] = scaler_data.transform(y_val[:, :, :, :args.input_base_dim])
            x_test[..., :args.input_base_dim] = scaler_data.transform(x_test[:, :, :, :args.input_base_dim])
            y_test[..., :args.input_base_dim] = scaler_data.transform(y_test[:, :, :, :args.input_base_dim])
        x_tra, y_tra = torch.FloatTensor(x_tra), torch.FloatTensor(y_tra)
        x_val, y_val = torch.FloatTensor(x_val), torch.FloatTensor(y_val)
        x_test, y_test = torch.FloatTensor(x_test), torch.FloatTensor(y_test)

        x_trn_dict[dataset_name], y_trn_dict[dataset_name] = x_tra, y_tra
        x_val_dict[dataset_name], y_val_dict[dataset_name] = x_val, y_val
        x_tst_dict[dataset_name], y_tst_dict[dataset_name] = x_test, y_test
        scaler_dict[dataset_name] = scaler_data

    return x_trn_dict, y_trn_dict, x_val_dict, y_val_dict, x_tst_dict, y_tst_dict, scaler_dict

def get_pretrain_task_batch(args, x_list, y_list):

    select_dataset = random.choice(args.dataset_use)
    print(args.dataset_use, select_dataset)
    batch_size = args.batch_size
    len_dataset = x_list[select_dataset].shape[0]

    batch_list_x = []
    batch_list_y = []
    permutation = np.random.permutation(len_dataset)
    for index in range(0, len_dataset, batch_size):
        start = index
        end = min(index + batch_size, len_dataset)
        indices = permutation[start:end]
        x_data = x_list[select_dataset][indices.copy()]
        y_data = y_list[select_dataset][indices.copy()]
        batch_list_x.append(x_data)
        batch_list_y.append(y_data)
    train_len = len(batch_list_x)
    return batch_list_x, batch_list_y, select_dataset, train_len

def get_val_tst_dataloader(X, Y, args, shuffle):
    X, Y = X[args.dataset_test], Y[args.dataset_test]
    data = torch.utils.data.TensorDataset(X, Y)
    data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=shuffle, drop_last=False)
    return data_loader

