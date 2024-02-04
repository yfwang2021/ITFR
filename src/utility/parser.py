import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run ")
    parser.add_argument('--data_path', nargs='?', default='./data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose the dataset')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=200,
                        help='Number of epoch.')
    parser.add_argument('--patience', type=int, default=200,
                        help='patience for early stop.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--regs', nargs='?', default='[0.0]',
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--Ks', nargs='?', default='[10,20,25]',
                        help='Top k(s) recommend')
    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')    
    parser.add_argument('--cuda_device', type=str, default="0",
                        help='cuda device')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')

    parser.add_argument('--rho', type=float, default=1, help='rho in sharpness-aware disadvantage group discovery')
    parser.add_argument('--eta', type=float, default=1, help='eta in collaborative loss balance')
    parser.add_argument('--gamma', type=float, default=1,
                        help='gamma in collaborative loss balance')
    parser.add_argument('--tau', type=float, default=1,
                        help='temperature for predicted score normalization')
    return parser.parse_args()