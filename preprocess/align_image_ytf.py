import sys
sys.path.insert(0, 'align_methods')
from align import align
import os
from tqdm import tqdm
from scipy.misc import imread, imsave
import cv2
import numpy as np

shapes = [(112, 112), (160, 160), (112, 96)]
data_dir = os.path.join('../', 'YouTubeFaces', 'frame_images_DB')
for name in tqdm(os.listdir(data_dir)):
    if '.' not in name:
        for video_id in os.listdir(os.path.join(data_dir, name)):
            filenames = os.listdir(os.path.join(data_dir, name, video_id))
            filenames.sort(key=lambda x:int(x.split('.')[1]))
            if name == 'Alastair_Campbell' and video_id == '2':
                path = os.path.join(data_dir, name, video_id, '2.1079.jpg')
            elif name == 'Choi_Sung-hong' and video_id == '2':
                path = os.path.join(data_dir, name, video_id, '2.3174.jpg')
            elif name == 'Claudia_Cardinale' and video_id == '1':
                path = os.path.join(data_dir, name, video_id, '1.961.jpg')
            elif name == 'Halle_Berry' and video_id == '0':
                path = os.path.join(data_dir, name, video_id, '0.282.jpg')
            elif name == 'Nathan_Lane' and video_id == '4':
                path = os.path.join(data_dir, name, video_id, '4.766.jpg')
            else:
                path = os.path.join(data_dir, name, video_id, filenames[len(filenames) // 2])
            img = imread(path).astype(np.float32)
            img = align(img)[0]
            img = img.astype(np.uint8)
            for shape in shapes:
                out = cv2.resize(img, (shape[1], shape[0]))
                output_dir = os.path.join('data', 'ytf-{}x{}'.format(shape[0], shape[1]), name)
                filename = '{}_{}.png'.format(name, video_id)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                imsave(os.path.join(output_dir, filename), out)
