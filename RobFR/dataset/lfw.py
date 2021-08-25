import os
import torch
import numpy as np
from RobFR.dataset.base import Loader
class LFWLoader(Loader):
    def __init__(self, datadir, goal, batch_size, model):
        super(LFWLoader, self).__init__(batch_size, model)
        with open(os.path.join('config', 'pairs_lfw.txt')) as f:
            lines = f.readlines() 
        suffix = '.jpg'
        self.pairs = []
        for line in lines:
            line = line.strip().split('\t')
            if len(line) == 3 and goal == 'dodging':
                path_src = os.path.join(datadir, line[0], line[0] + '_' + line[1].zfill(4) + suffix)
                path_dst = os.path.join(datadir, line[0], line[0] + '_' + line[2].zfill(4) + suffix)
                self.pairs.append([path_src, path_dst])
            elif len(line) == 4 and goal == 'impersonate':
                path_src = os.path.join(datadir, line[0], line[0] + '_' + line[1].zfill(4) + suffix)
                path_dst = os.path.join(datadir, line[2], line[2] + '_' + line[3].zfill(4) + suffix)
                self.pairs.append([path_src, path_dst])