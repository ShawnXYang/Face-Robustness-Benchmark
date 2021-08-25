import torch
import numpy as np
import torch.nn.functional as F
import random

def atanh(x):
    return 0.5*torch.log((1+x)/(1-x))
    
def Cutout(length, img_shape):
    h, w = img_shape
    mask = np.ones((h, w, 3), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = np.clip(y - length // 2, 0, h)
    y2 = np.clip(y + length // 2, 0, h)
    x1 = np.clip(x - length // 2, 0, w)
    x2 = np.clip(x + length // 2, 0, w)

    mask[y1: y2, x1: x2, :] = 0.
    return mask[None, :]

def Resize_and_Padding(x, scale_factor):
    h, w = x.shape[-2:]
    resized_h, resized_w = round(h * scale_factor), round(w * scale_factor)
    new_x = torch.zeros_like(x)
    offset_h, offset_w = random.randint(0, h - resized_h), random.randint(0, w - resized_w)
    new_x[:, :, offset_h:offset_h + resized_h, offset_w:offset_w + resized_w] = F.interpolate(x, (resized_h, resized_w))
    return new_x