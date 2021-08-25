from RobFR.attack.base import ConstrainedMethod
import torch
import os
from RobFR.attack.utils import Cutout

__all__ = ['CIM']
class CIM(ConstrainedMethod):
    def __init__(self, model, goal, distance_metric, eps, iters=20, mu=1.0, length=8):
        super(CIM, self).__init__(model, goal, distance_metric, eps)
        self.iters = iters
        self.mu = mu
        self.length = length
    def batch_attack(self, xs, ys_feat, **kwargs):
        xs_adv = xs.clone().detach().requires_grad_(True)
        g = torch.zeros_like(xs_adv)
        for _ in range(self.iters):
            img_shape = xs_adv.shape[2:]
            mask = Cutout(self.length, img_shape)
            mask = torch.Tensor(mask.transpose((0, 3, 1, 2))).cuda()
            features = self.model.forward(xs_adv * mask)
            loss = self.getLoss(features, ys_feat)
            loss.backward()
            grad = xs_adv.grad
            grad = grad / grad.abs().mean(dim=[1, 2, 3], keepdim=True)
            g = g * self.mu + grad
            self.model.zero_grad()
            xs_adv = self.step(xs_adv, 1.5 * self.eps / self.iters, g, xs, self.eps)
            xs_adv = xs_adv.detach().requires_grad_(True)
        return xs_adv

