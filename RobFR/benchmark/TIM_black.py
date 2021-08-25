import numpy as np
import torch
import os
import argparse

from RobFR.networks.get_model import getmodel
from RobFR.benchmark.utils import run_black
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
parser.add_argument('--batch_size', help='batch_size', type=int, default=20)
parser.add_argument('--kernel_len', help='length of kernel', type=int, default=7)
parser.add_argument('--nsig', help='sigma of kernel', type=float, default=3)
parser.add_argument('--distance', help='l2 or linf', type=str, default='linf', choices=['linf', 'l2'])
parser.add_argument('--num_samples', help='number of sampled landmarks', type=int, default=4)
parser.add_argument('--sigma', help='sigma', type=int, default=1)
parser.add_argument('--output', help='output dir', type=str, default='output/expdemo')
parser.add_argument('--use_lgc', action='store_true', default=False, help='whether to use lgc')
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def main():
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    model, img_shape = getmodel(args.model)

    attacker = attack.TIM(model=model,
        goal=args.goal, 
        distance_metric=args.distance, 
        eps=args.eps,
        mu=args.mu,
        num_samples=args.num_samples,
        iters=args.iters,
        sigma=args.sigma,
        use_lgc=args.use_lgc,
        dataset=args.dataset,
    )
        
    datadir = os.path.join('data', '{}-{}x{}'.format(args.dataset, img_shape[0], img_shape[1]))
    loader = LOADER_DICT[args.dataset](datadir, args.goal, args.batch_size, model)
    run_black(loader, attacker, args)

if __name__ == '__main__':
    main()

