import torch
import numpy as np

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
