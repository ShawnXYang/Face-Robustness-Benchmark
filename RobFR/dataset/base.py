from skimage.io import imread
import numpy as np
import os
import torch
def read_pair(path, device, model=None, return_feat=False):
    """
        Read image and get its logits
        Args:
            path: string, the path of image
            model: A FaceModel
        Returns:
            img: Tensor of shape (1, 3, H, W). HxW is the shape of input image. 160x160 for FaceNet, 112x96 for CosFace and SphereFace. 112x112 is the default shape.
            feat: Tensor of shape (1, model.out_dims). It is set to 512 in this paper.
    """
    img = imread(path).astype(np.float32)
    img = torch.Tensor(img.transpose((2, 0, 1))[None, :]).to(device)
    if not return_feat:
        return img
    feat = model.forward(img).detach().requires_grad_(False)
    return img, feat
class Loader:
    def __init__(self, batch_size, model):
        self.batch_size = batch_size
        self.model = model
        self.device = next(model.parameters()).device
        self.pairs = []
        self.pos = 0
    def __len__(self):
        return len(self.pairs) // self.batch_size
    def __iter__(self):
        return self
    def __next__(self):
        if self.pos < len(self.pairs):
            minibatches_pair = self.pairs[self.pos:self.pos+self.batch_size]
            self.pos += self.batch_size
            xs, ys, ys_feat = [], [], []
            for pair in minibatches_pair:
                path_src, path_dst = pair
                img_src = read_pair(path_src, self.device)
                img_dst, feat_dst = read_pair(path_dst, self.device, self.model, return_feat=True)
                xs.append(img_src)
                ys.append(img_dst)
                ys_feat.append(feat_dst)
            xs = torch.cat(xs)
            ys = torch.cat(ys)
            ys_feat = torch.cat(ys_feat)
            return xs, ys, ys_feat, minibatches_pair
        else:
            raise StopIteration

