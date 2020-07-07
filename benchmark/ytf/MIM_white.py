import numpy as np
import torch
import os
import argparse
from tqdm import tqdm

from networks.get_model import getmodel
from networks.config import threshold_ytf
from benchmark.ytf.utils import run_white, binsearch_alpha
import attack

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='White-box model', type=str, default='MobileFace', choices=threshold_ytf.keys())
parser.add_argument('--goal', help='dodging or impersonate', type=str, default='impersonate', choices=['dodging', 'impersonate'])
parser.add_argument('--eps', help='epsilon', type=float, default=16)
parser.add_argument('--iters', help='attack iteration', type=int, default=20)
parser.add_argument('--mu', help='momentum', type=float, default=1.0)
parser.add_argument('--seed', help='random seed', type=int, default=1234)
parser.add_argument('--steps', help='search steps', type=int, default=5)
parser.add_argument('--bin_steps', help='binary search steps', type=int, default=10)
parser.add_argument('--batch_size', help='batch_size', type=int, default=20)
parser.add_argument('--distance', help='l2 or linf', type=str, default='linf', choices=['linf', 'l2'])
parser.add_argument('--output', help='output dir', type=str, default='output/expdemo')
parser.add_argument('--log', help='log file', type=str, default='log.txt')
parser.add_argument('--save', action='store_true', default=True, help='whether to save images')
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def main():
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    model, img_shape = getmodel(args.model)
    config = {
        'eps': args.eps,
        'method': attack.MIM,
        'goal': args.goal,
        'distance_metric': args.distance,
        'threshold': threshold_ytf[args.model]['cos'],
        'steps': args.steps,
        'bin_steps': args.bin_steps,
        'model': model,
        'mu': args.mu,
        'iters': args.iters,
    }
        
    path = os.path.join('data', 'ytf-{}x{}'.format(img_shape[0], img_shape[1]))
    Attacker = lambda xs, ys, pairs: binsearch_alpha(xs=xs, ys=ys, pairs=pairs, **config)
    run_white(path, Attacker, model, args, threshold_ytf[args.model]['cos'])

if __name__ == '__main__':
    main()
