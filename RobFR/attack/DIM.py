from RobFR.attack.base import ConstrainedMethod
import torch
import os
from RobFR.attack.face_landmark import getlist_landmark
from RobFR.attack.utils import Resize_and_Padding
import random
import json

__all__ = ['DIM']
class DIM(ConstrainedMethod):
    def __init__(self, model, goal, distance_metric, eps, dataset='lfw',
        iters=20, mu=1.0, use_lgc=False, num_samples=4, sigma=1):
        super(DIM, self).__init__(model, goal, distance_metric, eps)
        self.iters = iters
        self.mu = mu
        self.num_samples = num_samples
        self.sigma = sigma
        self.use_lgc = use_lgc
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
            scale_factor = random.uniform(0.9, 1)
            img_shape = xs_adv.shape[2:]
            if self.use_lgc:
                mask = getlist_landmark(names, self.landmark_values,
                    self.num_samples, img_shape, sigma=self.sigma)
                mask = torch.Tensor(mask.transpose((0, 3, 1, 2))).cuda()
                features = self.model.forward(Resize_and_Padding(xs_adv * mask, scale_factor))
            else:
                features = self.model.forward(Resize_and_Padding(xs_adv, scale_factor))
            loss = self.getLoss(features, ys_feat)
            loss.backward()
            grad = xs_adv.grad
            grad = grad / grad.abs().mean(dim=[1, 2, 3], keepdim=True)
            g = g * self.mu + grad
            self.model.zero_grad()
            xs_adv = self.step(xs_adv, 1.5 * self.eps / self.iters, g, xs, self.eps)
            xs_adv = xs_adv.detach().requires_grad_(True)
        return xs_adv


if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 112).cuda()
    y = Resize_and_Padding(x, 0.8)
    print(y.shape)
