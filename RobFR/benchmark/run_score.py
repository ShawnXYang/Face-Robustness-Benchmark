from tqdm import tqdm
import argparse
import os

class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.avg = 0
		self.sum = 0
		self.cnt = 0

	def update(self, val, n=1):
		self.sum += val * n
		self.cnt += n
		self.avg = self.sum / self.cnt

parser = argparse.ArgumentParser()
parser.add_argument('--goal', help='dodging or impersonate', type=str, default='impersonate', choices=['dodging', 'impersonate'])
parser.add_argument('--log', help='log file', type=str, default='log.txt') 
args = parser.parse_args()

outputs = []
avg_score = AverageMeter()
avg_dist = AverageMeter()
avg_success = AverageMeter()
dists = []    
with open(os.path.join('log', args.log), 'r') as f:
    for line in f.readlines():
        adv_img_path, tar_img_path, score, dist, suc = line.strip().split(',')
        if score == 'score' or float(dist) == 0.0:
            continue
        score, dist, suc = float(score), float(dist), int(suc)
        avg_success.update(suc)
        avg_score.update(score)
        avg_dist.update(min(dist, 32))
        dists.append(dist)
print(avg_score.avg, avg_dist.avg, avg_success.avg)
dists.sort()
print(dists[len(dists) // 2])
