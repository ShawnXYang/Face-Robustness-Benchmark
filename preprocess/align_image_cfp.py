import sys
sys.path.insert(0, 'align_methods')
from align import align
import os
from tqdm import tqdm
from scipy.misc import imread, imsave
import cv2
import numpy as np

shapes = [(112, 112), (160, 160), (112, 96)]
data_dir = '/data/dingcheng/cfp-dataset/Data/Images/'
for name in tqdm(os.listdir(data_dir)):
    for image_type in os.listdir(os.path.join(data_dir, name)):
        image_dir = os.path.join(data_dir, name, image_type)
        for filename in os.listdir(image_dir):
            path = os.path.join(data_dir, name, image_type, filename)
            img = imread(path).astype(np.float32)
            img = align(img)[0]
            img = img.astype(np.uint8)
            for shape in shapes:
                out = cv2.resize(img, (shape[1], shape[0]))
                output_dir = os.path.join('data', 'cfp-{}x{}'.format(shape[0], shape[1]), name, image_type)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                imsave(os.path.join(output_dir, filename), out)
