import os
import sys
from networks.get_model import getmodel
import torch
import cv2
import numpy as np
from scipy import misc
import argparse
from benchmark.cfp.utils import read_pair, cosdistance
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='White-box model', type=str, default='MobileFace')
parser.add_argument('--log', help='log file', type=str, default='log/log.txt')
args = parser.parse_args()

config = {}
def get_paths(shape):
    pairs = []
    data_dir = '/data/dingcheng/cfp-dataset/'
    prefix = os.path.join(data_dir,'Protocol/')
    prefix_F = os.path.join(prefix, "Pair_list_F.txt")
    pairs_F = []
    prefix_P = os.path.join(prefix,"Pair_list_P.txt")
    pairs_P = []
    with open(prefix_F, 'r') as f:
        for line in f.readlines()[0:]:
            pair = line.strip().split()
            pairs_F.append(pair[1])
    with open(prefix_P, 'r') as f:
        for line in f.readlines()[0:]:
            pair = line.strip().split()
            pairs_P.append(pair[1])

    prefix = os.path.join(data_dir,"Protocol/Split/FP")
    folders_1 = os.listdir(prefix)
    for folder in folders_1:
        sublist = []
        same_list = []
        pairtxt = os.listdir(os.path.join(prefix, folder))
        for pair in pairtxt:
            img_root_path = os.path.join(prefix, folder, pair)
            with open(img_root_path, 'r') as f:
                for line in f.readlines()[0:]:
                    pair1 = line.strip().split(',')
                    id1, path1, filename1 = pairs_F[int(pair1[0])-1].split('/')[-3:]
                    path1 = os.path.join('data', 'cfp-{}x{}'.format(shape[0], shape[1]), id1, path1, filename1)
                    id2, path2, filename2 = pairs_P[int(pair1[1])-1].split('/')[-3:]
                    path2 = os.path.join('data', 'cfp-{}x{}'.format(shape[0], shape[1]), id2, path2, filename2)
                    if pair == 'same.txt':
                        pairs.append([path1, path2, 1])
                    else:
                        pairs.append([path1, path2, 0])
    return pairs
def read_pairs(path):
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
    for i in tqdm(range(num)):
        logits = []
        for filename in pairs[i][:2]:
            _, emb = read_pair(filename, model)
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
    print(args.model)
    model, shape = getmodel(args.model)
    pairs = get_paths(shape)
    test(pairs, model, shape, args.model) 
    with open(args.log, 'w') as f:
        for k, v in config.items():
            f.write('{}:{},\n'.format(k, v))

