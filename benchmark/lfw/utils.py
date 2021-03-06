import os
from scipy.misc import imread, imsave
import numpy as np
import torch
import attack
from tqdm import tqdm

def read_pair(path, model):
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
    img = torch.Tensor(img.transpose((2, 0, 1))[None, :]).cuda()
    feat = model.forward(img).detach().requires_grad_(False)
    return img, feat

def read_pairs(datadir, goal, model, batch_size):
    """
        Provide an iterable for LFW dataset.
        Args:
            datadir: The path of LFW dataset
            goal: impersonate or dodging
            model: A FaceModel
            batch_size: How many samples per batch to load
    """
    with open(os.path.join('data', 'pairs.txt')) as f:
        lines = f.readlines() 
    suffix = '.jpg'
    xs, ys, pairs = [], [], []
    idx = 0
    for line in lines:
        line = line.strip().split('\t')
        pair = []
        if len(line) == 3 and goal == 'dodging':
            path = os.path.join(datadir, line[0], line[0] + '_' + line[1].zfill(4) + suffix)
            pair.append(path)
            xs.append(read_pair(path, model)[0])
            path = os.path.join(datadir, line[0], line[0] + '_' + line[2].zfill(4) + suffix)
            pair.append(path)
            ys.append(read_pair(path, model)[1])
            pairs.append(pair)
        elif len(line) == 4 and goal == 'impersonate':
            path = os.path.join(datadir, line[0], line[0] + '_' + line[1].zfill(4) + suffix)
            pair.append(path)
            xs.append(read_pair(path, model)[0])
            path = os.path.join(datadir, line[2], line[2] + '_' + line[3].zfill(4) + suffix)
            pair.append(path)
            ys.append(read_pair(path, model)[1])
            pairs.append(pair)
        else:
            pass
        if len(ys) == batch_size:
            yield torch.cat(xs), torch.cat(ys), pairs
            xs, ys, pairs = [], [], []
    if len(ys) > 0:
        yield torch.cat(xs), torch.cat(ys), pairs

def save_images(image, original_image, filename, output_dir):
    """
        Save an adversarial example in png format.
        Args:
            image: An np.array. The adversarial example to be saved.
            original_image: An np.array. The original image.
            filename: string. The filename of saved image.
            output_dir: string. The directory of saved image.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image = np.clip(image, 0, 255).astype(np.uint8)
    imsave(os.path.join(output_dir, filename), image.astype(np.uint8), format='png')

def cosdistance(x, y, offset=1e-5):
    """
        The cosine distance of two vectors.
        Args:
            x: A tensor of shape (1, n)
            y: A tensor of shape (1, n)
    """
    x = x / torch.sqrt(torch.sum(x**2)) + offset
    y = y / torch.sqrt(torch.sum(y**2)) + offset
    return torch.sum(x * y)

def L2distance(x, y):
    """
        The L2 distance of two vectors.
        Args:
            x: A tensor of shape (1, n)
            y: A tensor of shape (1, n)
    """
    return torch.sqrt(torch.sum((x - y)**2))

def binsearch_basic(xs, ys, pairs, eps, method, threshold, steps=0, bin_steps=0, *args, **kwargs):
    """
        Use binary search to find an optimal eps of FGSM (White-box).
        Args:
            xs: A tensor of shape (batch, 3, H, W). Input images.
            ys: A tensor of shape (batch, m). m is set to 512 in this paper. The embedding vector of images.
            pairs: A list for each pair. Each pair contains (path1, path2), which means the path of victim image and target image.
            eps: float. The maximum perturbation.
            method: An Attack object, such as FGSM.
            threshold: float, The threshold of dodging or impersonate.
            steps: int. The number of linear search step.
            bin_steps, int. The number of binary search step.
        Returns:
            xs_results: A tensor of shape (batch, 3, H, W). The adversarial examples for xs.
    """
    batch_size = xs.size(0)
    lo = torch.zeros(batch_size, 1, 1, 1).cuda()
    hi = lo + eps
    eps_tensor = torch.zeros(batch_size, 1, 1, 1).cuda() + eps
    exp = 2 ** steps
    xs_results = xs.clone().detach()
    goal = kwargs['goal']
    model = kwargs['model']
# find a feasible eps by linear search
    for _ in range(exp):
        magnitude = (1.0 - float(_) / exp) * eps_tensor
        kwargs['eps'] = magnitude
        Attacker = method(**kwargs)
        xs_adv = Attacker.batch_attack(xs=xs, ys=ys, pairs=pairs)
        ys_adv = model.forward(xs_adv)
        similarities = torch.sum(ys_adv * ys, dim=1)
        if goal == 'dodging':
            succ_ = threshold - similarities > 0
        else:
            succ_ = similarities - threshold > 0
        xs_results[succ_] = xs_adv[succ_]
        hi[succ_] = (1.0 - float(_) / exp) * eps
    lo = hi - float(eps) / exp
# find an optimal eps by binary search
    for i in range(bin_steps):
        mi = (lo + hi) / 2
        kwargs['eps'] = mi
        Attacker = method(**kwargs)
        xs_adv = Attacker.batch_attack(xs=xs, ys=ys, pairs=pairs)
        ys_adv = model.forward(xs_adv)
        similarities = torch.sum(ys_adv * ys, dim=1)
        if goal == 'dodging':
            succ_ = threshold - similarities > 0
        else:
            succ_ = similarities - threshold > 0
        hi[succ_] = mi[succ_]
        lo[~succ_] = mi[~succ_]
        xs_results[succ_] = xs_adv[succ_]
    y = model.forward(xs)
    similarities = torch.sum(y * ys, dim=1)
    if goal == 'dodging':
        succ_ = threshold - similarities > 0
    else:
        succ_ = similarities - threshold > 0
    xs_results[succ_] = xs[succ_]
    return xs_results

def binsearch_alpha(xs, ys, pairs, eps, method, threshold, steps=0, bin_steps=0, *args, **kwargs):
    """
        Use binary search to find an optimal alpha of iterative gradient optimizer (White-box).
        Args:
            xs: A tensor of shape (batch, 3, H, W). Input images.
            ys: A tensor of shape (batch, m). m is set to 512 in this paper. The embedding vector of images.
            pairs: A list for each pair. Each pair contains (path1, path2), which means the path of victim image and target image.
            eps: float. The maximum perturbation.
            method: An Attack object, such as FGSM.
            threshold: float, The threshold of dodging or impersonate.
            steps: int. The number of linear search step.
            bin_steps, int. The number of binary search step.
        Returns:
            xs_results: A tensor of shape (batch, 3, H, W). The adversarial examples for xs.
    """
    batch_size = xs.size(0)
    lo = torch.zeros(batch_size, 1, 1, 1).cuda()
    hi = lo + eps
    xs_results = xs.clone().detach()
    goal = kwargs['goal']
    model = kwargs['model']
    found = torch.zeros(batch_size).bool().cuda()
# find a feasible alpha by linear search
    for _ in range(steps):
        kwargs['eps'] = hi
        Attacker = method(**kwargs)
        xs_adv = Attacker.batch_attack(xs=xs, ys=ys, pairs=pairs)
        ys_adv = model.forward(xs_adv)
        similarities = torch.sum(ys_adv * ys, dim=1)
        if goal == 'dodging':
            succ_ = threshold - similarities > 0
        else:
            succ_ = similarities - threshold > 0
        cond = ~found & succ_
        xs_results[cond] = xs_adv[cond]
        found[cond] = True
        not_found = ~found
        lo[not_found] = hi[not_found]
        hi[not_found] *= 2
        if found.all():
            break
# find an optimal alpha by binary search
    for i in range(bin_steps):
        mi = (lo + hi) / 2
        kwargs['eps'] = mi
        Attacker = method(**kwargs)
        xs_adv = Attacker.batch_attack(xs=xs, ys=ys, pairs=pairs)
        ys_adv = model.forward(xs_adv)
        similarities = torch.sum(ys_adv * ys, dim=1)
        if goal == 'dodging':
            succ_ = threshold - similarities > 0
        else:
            succ_ = similarities - threshold > 0
        hi[succ_] = mi[succ_]
        lo[~succ_] = mi[~succ_]
        xs_results[succ_] = xs_adv[succ_]
    y = model.forward(xs)
    similarities = torch.sum(y * ys, dim=1)
    if goal == 'dodging':
        succ_ = threshold - similarities > 0
    else:
        succ_ = similarities - threshold > 0
    xs_results[succ_] = xs[succ_]
    return xs_results

def run_black(path, Attacker, model, args):
    cnt = 0
    outputs = []
    for xs, ys, pairs in tqdm(read_pairs(path, args.goal, model, args.batch_size), total=3000//args.batch_size):
        x_adv = Attacker.batch_attack(xs=xs, ys=ys, pairs=pairs)
        for i in range(len(pairs)):
            img = x_adv[i].detach().cpu().numpy().transpose((1, 2, 0))
            cnt += 1
            npy_path = os.path.join(args.output, str(cnt) + '.npy')

            outputs.append([npy_path, pairs[i][0], pairs[i][1]])
            np.save(npy_path, img)
            if args.save:
                original_image = xs[i].cpu().numpy().transpose((1, 2, 0))
                save_images(img, original_image, str(cnt) + '.png', args.output)
    with open(os.path.join(args.output, 'annotation.txt'), 'w') as f:
        for pair in outputs:
            f.write('{} {} {}\n'.format(pair[0], pair[1], pair[2]))

def run_white(path, Attacker, model, args, threshold):
    cnt = 0
    scores = []
    dists = []
    success = []
    advs = []
    imgs = []
    for xs, ys, pairs in tqdm(read_pairs(path, args.goal, model, args.batch_size), total=3000//args.batch_size):
        x_adv = Attacker(xs=xs, ys=ys, pairs=pairs)
        y_adv = model.forward(x_adv)
        s = torch.sum(y_adv * ys, dim=1)
        for i in range(len(pairs)):
            img = x_adv[i].detach().cpu().numpy().transpose((1, 2, 0))
            x = xs[i].detach().cpu().numpy().transpose((1, 2, 0))
            scores.append(s[i].item())
            if args.goal == 'impersonate':
                success.append(int(s[i] > threshold))
            else:
                success.append(int(s[i] < threshold))
            cnt += 1
            advs.append(str(cnt) + '.npy')
            if args.distance == 'l2':
                dist = np.linalg.norm(img - x) / np.sqrt(img.reshape(-1).shape[0])
            else:
                dist = np.max(np.abs(img - x))
            dists.append(dist)
            imgs.append(pairs[i][1])
            if args.save:
                original_image = xs[i].cpu().numpy().transpose((1, 2, 0))
                save_images(img, original_image, str(cnt) + '.png', args.output)

    with open(os.path.join('log', args.log), 'w') as f:
        f.write('adv_img,tar_img,score,dist,success\n')
        for adv, img, score, d, s in zip(advs, imgs, scores, dists, success):
            f.write('{},{},{},{},{}\n'.format(adv, img, score, d, s))

