from RobFR.attack.base import ConstrainedMethod
import torch
import os
from RobFR.attack.face_landmark import getlist_landmark
import random
import numpy as np
import torch.nn.functional as F
import json

__all__ = ['TIM']
def smooth(stack_kernel, x):
    ''' implemenet depthwiseConv with padding_mode='SAME' in pytorch '''
    padding = (stack_kernel.size(-1) - 1) // 2
    groups = x.size(1)
    return F.conv2d(x, stack_kernel, padding=padding, groups=groups)
def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel
class TIM(ConstrainedMethod):
    def __init__(self, model, goal, distance_metric, eps, dataset='lfw',
        kernel_len=7, nsig=3, iters=20, mu=1.0, use_lgc=False, num_samples=4, sigma=1):
        super(TIM, self).__init__(model, goal, distance_metric, eps)
        self.iters = iters
        self.mu = mu
        self.num_samples = num_samples
        self.sigma = sigma
        self.use_lgc = use_lgc
        kernel = gkern(kernel_len, nsig).astype(np.float32)
        stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
        stack_kernel = np.expand_dims(stack_kernel, 3)
        stack_kernel = stack_kernel.transpose((2, 3, 0, 1))
        self.stack_kernel = torch.from_numpy(stack_kernel).cuda()
        with open('./data/{}_aligned_landmarks.json'.format(dataset), 'r') as f:
            self.landmark_values = json.load(f)
    def batch_attack(self, xs, ys_feat, pairs, **kwargs):
        xs_adv = xs.clone().detach().requires_grad_(True)
        names = []
        for pair in pairs:
            src_path = pair[0]
            tokens = src_path.split('/')
            name = '_'.join(tokens[3:])
            names.append(name)
        g = torch.zeros_like(xs_adv)
        for _ in range(self.iters):
            img_shape = xs_adv.shape[2:]
            if self.use_lgc:
                mask = getlist_landmark(names, self.landmark_values,
                    self.num_samples, img_shape, sigma=self.sigma)
                mask = torch.Tensor(mask.transpose((0, 3, 1, 2))).cuda()
                features = self.model.forward(xs_adv * mask)
            else:
                features = self.model.forward(xs_adv)
            loss = self.getLoss(features, ys_feat)
            loss.backward()
            grad = smooth(self.stack_kernel, xs_adv.grad)
            grad = grad / grad.abs().mean(dim=[1, 2, 3], keepdim=True)
            g = g * self.mu + grad
            self.model.zero_grad()
            xs_adv = self.step(xs_adv, 1.5 * self.eps / self.iters, g, xs, self.eps)
            xs_adv = xs_adv.detach().requires_grad_(True)
        return xs_adv
