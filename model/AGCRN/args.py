import argparse
import configparser

def parse_args(DATASET):
    config_file = '../conf/AGCRN/{}.conf'.format(DATASET)
    print('Read configuration file: %s' % (config_file))
    config = configparser.ConfigParser()
    config.read(config_file)
    print(config)
    #parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=DATASET, type=str)
    parser.add_argument('--device', default='cuda:0', type=str, help='indices of GPUs')
    parser.add_argument('--debug', default=False, type=eval)
    parser.add_argument('--cuda', default=True, type=bool)
    #data
    parser.add_argument('--lag', default=config['data']['lag'], type=int)
    parser.add_argument('--horizon', default=config['data']['horizon'], type=int)
    parser.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
    parser.add_argument('--tod', default=config['data']['tod'], type=eval)
    parser.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
    parser.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
    parser.add_argument('--default_graph', default=config['data']['default_graph'], type=eval)
    #model
    parser.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
    parser.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
    parser.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
    parser.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
    parser.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
    parser.add_argument('--cheb_k', default=config['model']['cheb_order'], type=int)
    args, _ = parser.parse_known_args()
    return args