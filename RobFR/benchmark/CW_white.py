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
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def main():
    model, img_shape = getmodel(args.model, device=args.device)
    attacker = attack.CW(model=model, goal=args.goal, 
        distance_metric=args.distance, 
        iteration=args.iters,
        threshold=THRESHOLD_DICT[args.dataset][args.model]['cos'],
        search_steps=args.steps,
        binsearch_steps=args.bin_steps,
        confidence=args.confidence,
        learning_rate=args.lr,
        c=args.c
    )
    datadir = os.path.join('data', '{}-{}x{}'.format(args.dataset, img_shape[0], img_shape[1]))
    loader = LOADER_DICT[args.dataset](datadir, args.goal, args.batch_size, model)
    Attacker = lambda xs, ys, ys_feat, pairs: attacker.batch_attack(xs=xs, ys=ys,
        ys_feat=ys_feat, pairs=pairs)
    run_white(loader, Attacker, model, args)

if __name__ == '__main__':
    main()

