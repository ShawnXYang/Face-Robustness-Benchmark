from RobFR.attack.base import ConstrainedMethod
import torch

__all__ = ['MIM']
class MIM(ConstrainedMethod):
    def __init__(self, model, goal, distance_metric, eps, iters=20, mu=1.0):
        super(MIM, self).__init__(model, goal, distance_metric, eps)
        self.iters = iters
        self.mu = mu
    def batch_attack(self, xs, ys_feat, **kwargs):
        xs_adv = xs.clone().detach().requires_grad_(True)
        g = torch.zeros_like(xs_adv)
        for _ in range(self.iters):
            features = self.model.forward(xs_adv)
            loss = self.getLoss(features, ys_feat)
            loss.backward()
            grad = xs_adv.grad
            grad = grad / grad.abs().mean(dim=[1, 2, 3], keepdim=True)
            g = g * self.mu + grad
            self.model.zero_grad()
            xs_adv = self.step(xs_adv, 1.5 * self.eps / self.iters, g, xs, self.eps)
            xs_adv = xs_adv.detach().requires_grad_(True)
        return xs_adv

