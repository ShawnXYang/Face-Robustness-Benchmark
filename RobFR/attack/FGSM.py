from RobFR.attack.base import ConstrainedMethod

__all__ = ['FGSM']
class FGSM(ConstrainedMethod):
    def __init__(self, model, goal, distance_metric, eps):
        super(FGSM, self).__init__(model, goal, distance_metric, eps)
    def batch_attack(self, xs, ys_feat, **kwargs):
        xs_adv = xs.clone().detach().requires_grad_(True)
        features = self.model.forward(xs_adv)
        loss = self.getLoss(features, ys_feat)
        loss.backward()
        grad = xs_adv.grad
        self.model.zero_grad()
        return self.step(xs_adv, self.eps, grad, xs, self.eps)
