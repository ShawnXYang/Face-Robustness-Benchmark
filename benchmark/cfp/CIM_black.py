import numpy as np
import torch
import os
import argparse

from networks.get_model import getmodel
from networks.config import threshold_cfp
from benchmark.cfp.utils import run_black
import attack

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='White-box model', type=str, default='MobileFace', choices=threshold_cfp.keys())
parser.add_argument('--goal', help='dodging or impersonate', type=str, default='impersonate', choices=['dodging', 'impersonate'])
parser.add_argument('--eps', help='epsilon', type=float, default=16)
parser.add_argument('--iters', help='attack iteration', type=int, default=20)
parser.add_argument('--mu', help='momentum', type=float, default=1.0)
parser.add_argument('--seed', help='random seed', type=int, default=1234)
parser.add_argument('--batch_size', help='batch_size', type=int, default=20)
parser.add_argument('--distance', help='l2 or linf', type=str, default='linf', choices=['linf', 'l2'])
parser.add_argument('--length', help='cutout length', type=int, default=10)
parser.add_argument('--output', help='output dir', type=str, default='output/expdemo')
parser.add_argument('--save', action='store_true', default=False, help='whether to save images')
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def main():
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    model, img_shape = getmodel(args.model)
    attacker = attack.CIM(model=model,
        goal=args.goal, 
        distance_metric=args.distance, 
        eps=args.eps,
        mu=args.mu,
        length=args.length,
        iters=args.iters
    )
        
        
    path = os.path.join('data', 'cfp-{}x{}'.format(img_shape[0], img_shape[1]))
    run_black(path, attacker, model, args)

if __name__ == '__main__':
    main()
