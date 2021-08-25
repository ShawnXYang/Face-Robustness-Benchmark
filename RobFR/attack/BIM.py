from RobFR.attack.base import ConstrainedMethod

__all__ = ['BIM']
class BIM(ConstrainedMethod):
    def __init__(self, model, goal, distance_metric, eps, iters=20):
        super(BIM, self).__init__(model, goal, distance_metric, eps)
        self.iters = iters
    def batch_attack(self, xs, ys_feat, **kwargs):
        xs_adv = xs.clone().detach().requires_grad_(True)
        for _ in range(self.iters):
            features = self.model.forward(xs_adv)
            loss = self.getLoss(features, ys_feat)
            loss.backward()
            grad = xs_adv.grad
            self.model.zero_grad()
            xs_adv = self.step(xs_adv, 1.5 * self.eps / self.iters, grad, xs, self.eps)
            xs_adv = xs_adv.detach().requires_grad_(True)
        return xs_adv
