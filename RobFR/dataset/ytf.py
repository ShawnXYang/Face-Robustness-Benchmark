import os
import torch
import numpy as np
from RobFR.dataset.base import Loader
class YTFLoader(Loader):
    def __init__(self, datadir, goal, batch_size, model):
        super(YTFLoader, self).__init__(batch_size, model)
        with open(os.path.join('config', 'pairs_ytf.txt')) as f:
            lines = f.readlines() 
        suffix = '.jpg'
        self.pairs = []
        for line in lines:
            line = line.strip().replace(' ', '').split(',')
            if line[0] == 'splitnumber':
                continue
            if (int(line[-1]) == 1 and goal == 'dodging') or (int(line[-1]) == 0 and goal == 'impersonate'):
                pair = []
                name, pid = line[2].split('/')
                path_src = os.path.join(datadir, name, name + '_' + pid + '.png')
                name, pid = line[3].split('/')
                path_dst = os.path.join(datadir, name, name + '_' + pid + '.png')
                self.pairs.append([path_src, path_dst])