import sys
sys.path.insert(0, 'align_methods')
from align import align
import os
from tqdm import tqdm
from scipy.misc import imread, imsave
import cv2
import numpy as np

shapes = [(112, 112), (160, 160), (112, 96)]
data_dir = os.path.join('data', 'lfw')
for name in tqdm(os.listdir(data_dir)):
    if 'txt' not in name:
        for filename in os.listdir(os.path.join(data_dir, name)):
            img = imread(os.path.join(data_dir, name, filename)).astype(np.float32)
            img = align(img)[0]
            img = img.astype(np.uint8)
            for shape in shapes:
                out = cv2.resize(img, (shape[1], shape[0]))
                output_dir = os.path.join('data', 'lfw-{}x{}'.format(shape[0], shape[1]), name)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                imsave(os.path.join(output_dir, filename), out)
