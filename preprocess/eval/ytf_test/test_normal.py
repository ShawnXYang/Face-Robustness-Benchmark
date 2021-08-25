import os
import sys
from networks.get_model import getmodel
import torch
import cv2
import numpy as np
from scipy import misc
import argparse
from benchmark.ytf.utils import read_pair, cosdistance
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='White-box model', type=str, default='MobileFace')
parser.add_argument('--log', help='log file', type=str, default='log/log.txt')
args = parser.parse_args()

config = {}
def read_pairs(path):
    with open(os.path.join(path, 'splits.txt')) as f:
        lines = f.readlines() 
    suffix = '.png'
    pairs = []
    for line in lines:
        line = line.strip().replace(' ', '').split(',')
        if line[0] == 'splitnumber':
            continue
        pair = []
        name, pid = line[2].split('/')
        pair.append(os.path.join(name, name + '_' + pid + suffix))
        name, pid = line[3].split('/')
        pair.append(os.path.join(name, name + '_' + pid + suffix))
        pair.append(int(line[-1]))
        pairs.append(pair)
    return pairs


def test(pairs, model, shape, model_name):
    num = len(pairs)
    result_cos = np.empty(shape=(num))
    gt = np.empty(shape=(num))
    
    zero_matrix = np.zeros(shape=(1, shape[0], shape[1], 3))
    data_path = os.path.join('data', 'ytf-{}x{}'.format(shape[0], shape[1]))

    for i in tqdm(range(num)):
        logits = []
        for filename in pairs[i][:2]:
            _, emb = read_pair(os.path.join(data_path, filename), model)
            logits.append(emb)
        result_cos[i] = cosdistance(logits[0], logits[1]).item()
        gt[i] = pairs[i][2]
    cos_best_value = 0
    for x in result_cos:
        acc = (result_cos > x) == gt
        acc = acc.sum() * 1.0 / num
        if cos_best_value < acc:
            cos_best_value = acc
            cos_best_threshold = x
    print('cos distance: ', cos_best_value, cos_best_threshold, model_name)
    config[model_name] = {
        'cos': cos_best_threshold,
        'cos_acc': cos_best_value,
    }

if __name__ == '__main__':
    dst = os.path.join('./data')
    pairs = read_pairs(dst)
    print(args.model)
    model, shape = getmodel(args.model)
    test(pairs, model, shape, args.model) 
    with open(args.log, 'w') as f:
        for k, v in config.items():
            f.write('{}:{},\n'.format(k, v))

