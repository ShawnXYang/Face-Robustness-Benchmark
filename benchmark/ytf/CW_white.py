import numpy as np
import torch
import os
import argparse
from tqdm import tqdm

from networks.get_model import getmodel
from networks.config import threshold_ytf
from benchmark.ytf.utils import run_white
import attack

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='White-box model', type=str, default='MobileFace', choices=threshold_ytf.keys())
parser.add_argument('--goal', help='dodging or impersonate', type=str, default='impersonate', choices=['dodging', 'impersonate'])
parser.add_argument('--iters', help='attack iteration', type=int, default=100)
parser.add_argument('--steps', help='search steps', type=int, default=10)
parser.add_argument('--bin_steps', help='binary search steps', type=int, default=10)
parser.add_argument('--confidence', help='kappa', type=float, default=1e-3)
parser.add_argument('--c', help='c', type=float, default=1)
parser.add_argument('--eps', help='epsilon', type=float, default=255)
parser.add_argument('--lr', help='learning rate', type=float, default=1e-3)
parser.add_argument('--seed', help='random seed', type=int, default=1234)
parser.add_argument('--distance', help='l2', type=str, default='l2', choices=['l2'])
parser.add_argument('--batch_size', help='batch_size', type=int, default=20)
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
    attacker = attack.CW(model=model,
        goal=args.goal, 
        distance_metric=args.distance, 
        iteration=args.iters,
        threshold=threshold_ytf[args.model]['cos'],
        search_steps=args.steps,
        binsearch_steps=args.bin_steps,
        confidence=args.confidence,
        learning_rate=args.lr,
        c=args.c
    )
    path = os.path.join('data', 'ytf-{}x{}'.format(img_shape[0], img_shape[1]))
    Attacker = lambda xs, ys, pairs: attacker.batch_attack(xs, ys, pairs=pairs)
    run_white(path, Attacker, model, args, threshold_ytf[args.model]['cos'])

if __name__ == '__main__':
    main()

