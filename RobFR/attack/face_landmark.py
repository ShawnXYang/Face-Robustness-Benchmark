from __future__ import print_function
import os
import sys

import time
import math
import numpy as np
import json
import random
import cv2


def get_sample_landmark(name, landmark_values, num_samples = 4, sigma=1):
    pts_img = []
    for pts in landmark_values[name]:
        if pts[0] > 0 and pts[0] < 112 and pts[1] < 112:
            pts_img.append(pts)
    pts_img = np.array(pts_img)
    heatmaps = np.zeros((112, 112), dtype=np.float32)
    sample_list = random.sample(list(np.arange(17,len(pts_img))), min(num_samples, len(pts_img) - 17))
    
    for i in sample_list:
        heatmaps = draw_gaussian(heatmaps, pts_img[i], sigma)
    heatmaps = np.tile(heatmaps[..., np.newaxis], (1, 1, 3))
    return heatmaps
    
def getlist_landmark(names, landmark_values, num_samples=4, img_shape=(112,112), sigma=1):
    lands = np.zeros(shape = (len(names),) + img_shape + (3,), dtype=np.float32)
    for idx, name in enumerate(names):
        land = get_sample_landmark(name, landmark_values, num_samples, sigma)
        land = cv2.resize(land, (img_shape[1], img_shape[0]))
        lands[idx] = land == 0
    return lands
        
def _gaussian(
        size=3, sigma=0.25, amplitude=1, normalize=False, width=None,
        height=None, sigma_horz=None, sigma_vert=None, mean_horz=0.5,
        mean_vert=0.5):
    # handle some defaults
    if width is None:
        width = size
    if height is None:
        height = size
    if sigma_horz is None:
        sigma_horz = sigma
    if sigma_vert is None:
        sigma_vert = sigma
    center_x = mean_horz * width + 0.5
    center_y = mean_vert * height + 0.5
    gauss = np.empty((height, width), dtype=np.float32)
    # generate kernel
    for i in range(height):
        for j in range(width):
            gauss[i][j] = amplitude * math.exp(-(math.pow((j + 1 - center_x) / (
                sigma_horz * width), 2) / 2.0 + math.pow((i + 1 - center_y) / (sigma_vert * height), 2) / 2.0))
    if normalize:
        gauss = gauss / np.sum(gauss)
    return gauss

def draw_gaussian(image, point, sigma):
    # Check if the gaussian is inside
    ul = [math.floor(point[0] - 3 * sigma), math.floor(point[1] - 3 * sigma)]
    br = [math.floor(point[0] + 3 * sigma), math.floor(point[1] + 3 * sigma)]
    ul = [math.floor(point[0] - 1 * sigma), math.floor(point[1] - 1 * sigma)]
    br = [math.floor(point[0] + 1 * sigma), math.floor(point[1] + 1 * sigma)]
    if (ul[0] > image.shape[1] or ul[1] > image.shape[0] or br[0] < 1 or br[1] < 1):
        return image
    size = 6 * sigma + 1
    size = 2 * sigma + 1
    g = _gaussian(size)
    g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) - int(max(1, ul[0])) + int(max(1, -ul[0]))]
    g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) - int(max(1, ul[1])) + int(max(1, -ul[1]))]
    img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
    img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
    assert (g_x[0] > 0 and g_y[1] > 0)
    image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]
          ] = image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] + g[g_y[0] - 1:g_y[1], g_x[0] - 1:g_x[1]]
    image[image > 1] = 1
    return image
