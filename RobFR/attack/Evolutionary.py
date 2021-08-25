from RobFR.attack.base import Attack
import torch
import torch.nn.functional as F
import numpy as np
from skimage.io import imread
import cv2

__all__ = ['Evolutionary']
class Evolutionary(Attack):
    def __init__(self, model, goal, distance_metric, threshold, c=0.001, decay_weight=0.99, max_queries=10000, mu=0.01, sigma=3e-2, freq=30):
        assert(distance_metric == 'l2')
        super(Evolutionary, self).__init__(model, goal, distance_metric)
        self.threshold = threshold
        self.c = c
        self.decay_weight = decay_weight
        self.max_queries = max_queries
        self.mu = mu
        self.sigma = sigma
        self.freq = freq


    def batch_attack(self, xs, ys_feat, ys, **kwargs):
        adv = torch.zeros_like(xs)
        for i in range(xs.size(0)):
            x_target = ys[i]
            adv[i] = self.attack(xs[None, i], ys_feat[i], x_target, self.model)
        return adv
    def attack(self, x, y, x_target, model):
        pert_shape = (32, 32, 3)
        N = np.prod(pert_shape)
        K = N // 20
        evolutionary_path = np.zeros(pert_shape)
        decay_weight = self.decay_weight
        diagnoal_covariance = np.ones(pert_shape)
        c = self.c
# find an starting point
        if self.goal == 'impersonate':
            x_adv = x_target
        else:
            while True:
                x_adv = torch.rand(x.shape).cuda() * 255
                y_adv = model.forward(x_adv)
                similarity = torch.sum(y * y_adv)
                if similarity < self.threshold:
                    break
            start = x.clone()
            end = x_adv.clone()
            for s in range(10):
                interpolated = (start + end) / 2
                y_adv = model.forward(interpolated)
                similarity = torch.sum(y * y_adv)
                if similarity < self.threshold:
                    end = interpolated
                else:
                    start = interpolated
            x_adv = end
            
        mindist = 1e10
        freq = self.freq
        mu = self.mu
        stats_adversarial = []
        for _ in range(self.max_queries):
            unnormalized_source_direction = x - x_adv
            source_norm = torch.norm(unnormalized_source_direction)
            if mindist > source_norm:
                mindist = source_norm
                best_adv = x_adv
            selection_prob = diagnoal_covariance.reshape(-1) / np.sum(diagnoal_covariance)
            selection_indices = np.random.choice(N, K, replace=False, p=selection_prob)
            pert = np.random.normal(0, 1, pert_shape)
            factor = np.zeros(N)
            factor[selection_indices] = True
            pert *= factor.reshape(pert_shape) * np.sqrt(diagnoal_covariance)
            pert_large = cv2.resize(pert, x.shape[2:])
            pert_large = torch.Tensor(pert_large[None, :]).cuda().permute(0, 3, 1, 2)
            biased = x_adv + mu * unnormalized_source_direction
            candidate = biased + self.sigma * source_norm * pert_large / torch.norm(pert_large)
            candidate = x - (x - candidate) / torch.norm(x - candidate) * torch.norm(x - biased)
            candidate = torch.clamp(candidate, min=0, max=255)
            y_adv = model.forward(candidate)
            similarity = torch.sum(y_adv * y)
            if (self.goal == 'dodging' and similarity < self.threshold) or (self.goal == 'impersonate' and similarity > self.threshold):
                x_adv = candidate
                evolutionary_path = decay_weight * evolutionary_path + np.sqrt(1 - decay_weight ** 2) * pert
                diagnoal_covariance = (1 - c) * diagnoal_covariance + c * (evolutionary_path ** 2)
                stats_adversarial.append(1)
            else:
                stats_adversarial.append(0)
            if len(stats_adversarial) == 30:
                p_step = np.mean(stats_adversarial)
                mu *= np.exp(p_step - 0.2)
                stats_adversarial = []
        return best_adv
