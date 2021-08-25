import os
import torch
import numpy as np
from RobFR.dataset.base import Loader
class CFPLoader(Loader):
    def __init__(self, datadir, goal, batch_size, model):
        super(CFPLoader, self).__init__(batch_size, model)
        self.pairs = []
        prefix = os.path.join('config', 'Protocol/')
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
        prefix = os.path.join('config', "Protocol/Split/FP")
        folders_1 = os.listdir(prefix)
        for folder in folders_1:
            if goal == 'dodging':
                pair = 'same.txt'
            else:
                pair = 'diff.txt'
            img_root_path = os.path.join(prefix, folder, pair)
            with open(img_root_path, 'r') as f:
                for line in f.readlines()[0:]:
                    pair1 = line.strip().split(',')
                    id1, path1, filename1 = pairs_F[int(pair1[0])-1].split('/')[-3:]
                    path_src = os.path.join(datadir, id1, path1, filename1)
                    id2, path2, filename2 = pairs_P[int(pair1[1])-1].split('/')[-3:]
                    path_dst = os.path.join(datadir, id2, path2, filename2)
                    self.pairs.append([path_src, path_dst])
