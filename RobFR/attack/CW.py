from RobFR.attack.base import Attack
from RobFR.attack.utils import atanh
import torch
import torch.nn.functional as F
import numpy as np

__all__ = ['CW']
class CW(Attack):
    def __init__(self, model, goal, distance_metric, threshold, iteration=100, search_steps=6, binsearch_steps=10, confidence=1e-3, learning_rate=1e-3, c=1e-3):
        assert(distance_metric == 'l2')
        super(CW, self).__init__(model, goal, distance_metric)
        self.iteration = iteration
        self.search_steps = search_steps
        self.binsearch_steps = binsearch_steps
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.c = c


    def batch_attack(self, xs, ys_feat, **kwargs):
        batch_size = xs.size(0)
        cs = torch.ones(batch_size).cuda() * self.c
        xs_adv = torch.zeros_like(xs)
        min_dists = torch.zeros(batch_size).fill_(np.inf).cuda()
        found = torch.zeros(batch_size).bool().cuda()
        for _ in range(self.search_steps):
            ws = atanh((xs - 127.5) / 127.5).detach().requires_grad_(True)
            optimizer = torch.optim.Adam([ws], lr=self.learning_rate)
            for _ in range(self.iteration):
                xs_adv_ = 255 * 0.5 * (torch.tanh(ws) + 1)
                logits = self.model.forward(xs_adv_)
                similarities = torch.sum(logits * ys_feat, dim=1)
                dists = torch.sum((xs_adv_.sub(xs).div(255)) ** 2, dim=[1, 2, 3])
                if self.goal == 'dodging':
                    scores = torch.clamp(similarities - self.threshold + self.confidence, min=0)
                    succ_ = self.threshold - similarities > self.confidence
                else:
                    scores = torch.clamp(self.threshold - similarities + self.confidence, min=0)
                    succ_ = similarities - self.threshold > self.confidence
                loss = torch.sum(dists + scores * cs)
                better_dists = dists < min_dists
                to_update = succ_ & better_dists
                xs_adv[to_update] = xs_adv_[to_update]
                min_dists[to_update] = dists[to_update]
                found[to_update] = True
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if found.all():
                break
            else:
                cs[~found] *= 10
        cs_hi = cs
        cs_lo = torch.zeros_like(cs)
        cs = (cs_hi + cs_lo) / 2
        for _ in range(self.binsearch_steps):
            ws = torch.nn.Parameter(atanh((xs - 127.5) / 127.5).detach(), requires_grad=True)
            optimizer = torch.optim.Adam([ws], lr=self.learning_rate)
            succ = torch.zeros(batch_size).bool().cuda()
            for _ in range(self.iteration):
                xs_adv_ = 255 * 0.5 * (torch.tanh(ws) + 1)
                logits = self.model.forward(xs_adv_)
                similarities = torch.sum(logits * ys_feat, dim=1)
                dists = torch.sum((xs_adv_.sub(xs).div(255)) ** 2, dim=[1, 2, 3])
                if self.goal == 'dodging':
                    scores = torch.clamp(similarities - self.threshold + self.confidence, min=0)
                    succ_ = self.threshold - similarities > self.confidence
                else:
                    scores = torch.clamp(self.threshold - similarities + self.confidence, min=0)
                    succ_ = similarities - self.threshold > self.confidence
                loss = torch.sum(dists + scores * cs)
                better_dists = dists < min_dists
                to_update = succ_ & better_dists
                succ[to_update] = True
                xs_adv[to_update] = xs_adv_[to_update]
                found[to_update] = True
                min_dists[to_update] = dists[to_update]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            cs_hi[succ] = cs[succ]
            cs_lo[~succ] = cs[~succ]
            cs = (cs_hi + cs_lo) / 2
        return xs_adv, found
