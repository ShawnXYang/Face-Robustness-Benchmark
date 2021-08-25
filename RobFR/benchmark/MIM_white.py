import numpy as np
import torch
import os
import argparse
from tqdm import tqdm

from RobFR.networks.get_model import getmodel
from RobFR.networks.config import THRESHOLD_DICT
from RobFR.benchmark.utils import binsearch_alpha, run_white, binsearch_basic
from RobFR.dataset import LOADER_DICT
import RobFR.attack as attack

parser = argparse.ArgumentParser()
parser.add_argument('--device', help='device id', type=str, default='cuda')
parser.add_argument('--dataset', help='dataset', type=str, default='lfw', choices=['lfw', 'ytf', 'cfp'])
parser.add_argument('--model', help='White-box model', type=str, default='MobileFace')
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
    model, img_shape = getmodel(args.model, device=args.device)
    config = {
        'eps': args.eps,
        'method': attack.MIM,
        'goal': args.goal,
        'distance_metric': args.distance,
        'threshold': THRESHOLD_DICT[args.dataset][args.model]['cos'],
        'steps': args.steps,
        'bin_steps': args.bin_steps,
        'model': model,
        'mu': args.mu,
        'iters': args.iters,
    }
    datadir = os.path.join('data', '{}-{}x{}'.format(args.dataset, img_shape[0], img_shape[1]))
    loader = LOADER_DICT[args.dataset](datadir, args.goal, args.batch_size, model)
    Attacker = lambda xs, ys, ys_feat, pairs: binsearch_alpha(xs=xs, ys=ys, 
        ys_feat=ys_feat, pairs=pairs, **config)
    run_white(loader, Attacker, model, args)
        

if __name__ == '__main__':
    main()
