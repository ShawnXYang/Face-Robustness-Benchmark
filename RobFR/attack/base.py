import torch
import numpy as np
class Attack(object):
    def __init__(self, model, goal, distance_metric):
        assert(goal == 'dodging' or goal == 'impersonate')
        assert(distance_metric == 'linf' or distance_metric == 'l2')
        self.model = model
        self.goal = goal
        self.distance_metric = distance_metric
class ConstrainedMethod(Attack):
    def __init__(self, model, goal, distance_metric, eps):
        super(ConstrainedMethod, self).__init__(model, goal, distance_metric)
        self.eps = eps
    def getLoss(self, features, ys):
        if self.goal == 'impersonate':
            return torch.mean((ys - features) ** 2)
        else:
            return -torch.mean((ys - features) ** 2)
    def clip_by_value(self, xs_adv, xs, eps):
        minx = torch.clamp(xs - eps, min=0)
        maxx = torch.clamp(xs + eps, max=255)
        xs_adv = torch.min(xs_adv, maxx)
        xs_adv = torch.max(xs_adv, minx)
        return xs_adv
    def clip_by_norm(self, xs_adv, xs, r):
        delta = xs_adv - xs
        batch_size = delta.size(0)
        delta_2d = delta.view(batch_size, -1)
        if isinstance(r, torch.Tensor):
            delta_norm = torch.max(torch.norm(delta_2d, dim=1), r.view(-1)).view(batch_size, 1, 1, 1)
        else:
            delta_norm = torch.clamp(torch.norm(delta_2d, dim=1), min=r).view(batch_size, 1, 1, 1)
        factor = r / delta_norm
        return xs + delta * factor
    def step(self, xs_adv, lr, grad, xs, eps):
        if self.distance_metric == 'linf':
            xs_adv = xs_adv - lr * torch.sign(grad)
            xs_adv = self.clip_by_value(xs_adv, xs, eps)
        else:
            batch_size = grad.size(0)
            grad_2d = grad.view(batch_size, -1)
            grad_norm = torch.clamp(torch.norm(grad_2d, dim=1), min=1e-12).view(batch_size, 1, 1, 1)
            grad_unit = grad / grad_norm
            alpha = lr * np.sqrt(grad[0].numel())
            xs_adv = xs_adv - alpha * grad_unit
            xs_adv = self.clip_by_norm(xs_adv, xs, eps * np.sqrt(grad[0].numel()))
        return torch.clamp(xs_adv, min=0, max=255)
