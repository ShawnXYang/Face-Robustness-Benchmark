from RobFR.benchmark.utils import cosdistance
from networks.get_model import getmodel
from tqdm import tqdm
import argparse
import torch
import os
import numpy as np
import cv2
from skimage.io import imread
from RobFR.networks.config import THRESHOLD_DICT

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='dataset', type=str, default='lfw', choices=['lfw', 'ytf', 'cfp'])
parser.add_argument('--model', help='White-box model', type=str, default='MobileFace')
parser.add_argument('--goal', help='dodging or impersonate', type=str, default='impersonate', choices=['dodging', 'impersonate'])
parser.add_argument('--distance', help='l2 or linf', type=str, default='linf', choices=['linf', 'l2'])
parser.add_argument('--anno', help='output dir', type=str, default='output/expdemo/annotation.txt')
parser.add_argument('--log', help='log file', type=str, default='log.txt') 
args = parser.parse_args()

model, img_shape = getmodel(args.model)
outputs = []
threshold = THRESHOLD_DICT[args.dataset][args.model]['cos']
with open(args.anno, 'r') as f:
    for line in tqdm(f.readlines()):
        adv_img_path, src_img_path, tar_img_path = line.strip().split(' ')

        adv_img = np.load(adv_img_path)
        resized_adv_img = cv2.resize(adv_img, (img_shape[1], img_shape[0])) 
        resized_adv_img = torch.Tensor(resized_adv_img.transpose(2, 0, 1)[None, :]).cuda()
        adv_feat = model.forward(resized_adv_img)
    
        src_img = imread(src_img_path).astype(np.float32)

        tar_img = imread(tar_img_path).astype(np.float32)
        resized_tar_img = cv2.resize(tar_img, (img_shape[1], img_shape[0]))
        resized_tar_img = torch.Tensor(resized_tar_img.transpose((2, 0, 1))[None, :]).cuda()
        tar_feat = model.forward(resized_tar_img)

        score = cosdistance(adv_feat, tar_feat).item()
        if args.goal == 'dodging':
            suc = int(score < threshold)
        else:
            suc = int(score > threshold)
        if args.distance == 'l2':
            dist = np.linalg.norm(adv_img - src_img) / np.sqrt(adv_img.reshape(-1).shape[0])
        else:
            dist = np.max(np.abs(adv_img - src_img))
        outputs.append([os.path.basename(adv_img_path), os.path.basename(tar_img_path), score, dist, suc])
with open(os.path.join('log', args.log), 'w') as f:
    f.write('adv_img,tar_img,score,dist,success\n')
    for output in outputs:
        f.write('{},{},{},{},{}\n'.format(output[0], output[1], output[2], output[3], output[4]))
