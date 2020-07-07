from attack.base import ConstrainedMethod
import torch
import os
from attack.face_landmark import getlist_landmark

__all__ = ['LGC']
class LGC(ConstrainedMethod):
    def __init__(self, model, goal, distance_metric, eps, iters=20, mu=1.0, num_samples=4, sigma=1):
        super(LGC, self).__init__(model, goal, distance_metric, eps)
        self.iters = iters
        self.mu = mu
        self.num_samples = num_samples
        self.sigma = sigma
    def batch_attack(self, xs, ys, pairs, **kwargs):
        xs_adv = xs.clone().detach().requires_grad_(True)
        names = [os.path.basename(x[0])[:-4] for x in pairs]
#        names = ['-'.join(img_path[0].split('/')[-3:]) for img_path in pairs]
        g = torch.zeros_like(xs_adv)
        for _ in range(self.iters):
            img_shape = xs_adv.shape[2:]
            mask = getlist_landmark(names, self.num_samples, img_shape, sigma=self.sigma)
            mask = torch.Tensor(mask.transpose((0, 3, 1, 2))).cuda()
            features = self.model.forward(xs_adv * mask)
            loss = self.getLoss(features, ys)
            loss.backward()
            grad = xs_adv.grad
            grad = grad / grad.abs().mean(dim=[1, 2, 3], keepdim=True)
            g = g * self.mu + grad
            self.model.zero_grad()
            xs_adv = self.step(xs_adv, 1.5 * self.eps / self.iters, g, xs, self.eps)
            xs_adv = xs_adv.detach().requires_grad_(True)
        return xs_adv

