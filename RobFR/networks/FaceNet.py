import torch
from torch import nn
from torch.nn import functional as F
import os
from RobFR.networks.FaceModel import FaceModel

def prewhiten(x):
	mean = x.mean()
	std = x.std()
	std_adj = std.clamp(min=1.0/(float(x.numel())**0.5))
	y = (x - mean) / std_adj
	return y

class BasicConv2d(nn.Module):

	def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
		super().__init__()
		self.conv = nn.Conv2d(
			in_planes, out_planes,
			kernel_size=kernel_size, stride=stride,
			padding=padding, bias=False
		) # verify bias false
		self.bn = nn.BatchNorm2d(
			out_planes,
			eps=0.001, # value found in tensorflow
			momentum=0.1, # default pytorch value
			affine=True
		)
		self.relu = nn.ReLU(inplace=False)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = self.relu(x)
		return x


class Block35(nn.Module):
	
	def __init__(self, scale=1.0):
		super().__init__()

		self.scale = scale

		self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)

		self.branch1 = nn.Sequential(
			BasicConv2d(256, 32, kernel_size=1, stride=1),
			BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
		)

		self.branch2 = nn.Sequential(
			BasicConv2d(256, 32, kernel_size=1, stride=1),
			BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
			BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
		)

		self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
		self.relu = nn.ReLU(inplace=False)

	def forward(self, x):
		x0 = self.branch0(x)
		x1 = self.branch1(x)
		x2 = self.branch2(x)
		out = torch.cat((x0, x1, x2), 1)
		out = self.conv2d(out)
		out = out * self.scale + x
		out = self.relu(out)
		return out


class Block17(nn.Module):

	def __init__(self, scale=1.0):
		super().__init__()

		self.scale = scale

		self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)

		self.branch1 = nn.Sequential(
			BasicConv2d(896, 128, kernel_size=1, stride=1),
			BasicConv2d(128, 128, kernel_size=(1,7), stride=1, padding=(0,3)),
			BasicConv2d(128, 128, kernel_size=(7,1), stride=1, padding=(3,0))
		)

		self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
		self.relu = nn.ReLU(inplace=False)

	def forward(self, x):
		x0 = self.branch0(x)
		x1 = self.branch1(x)
		out = torch.cat((x0, x1), 1)
		out = self.conv2d(out)
		out = out * self.scale + x
		out = self.relu(out)
		return out


class Block8(nn.Module):

	def __init__(self, scale=1.0, noReLU=False):
		super().__init__()

		self.scale = scale
		self.noReLU = noReLU

		self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)

		self.branch1 = nn.Sequential(
			BasicConv2d(1792, 192, kernel_size=1, stride=1),
			BasicConv2d(192, 192, kernel_size=(1,3), stride=1, padding=(0,1)),
			BasicConv2d(192, 192, kernel_size=(3,1), stride=1, padding=(1,0))
		)

		self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
		if not self.noReLU:
			self.relu = nn.ReLU(inplace=False)

	def forward(self, x):
		x0 = self.branch0(x)
		x1 = self.branch1(x)
		out = torch.cat((x0, x1), 1)
		out = self.conv2d(out)
		out = out * self.scale + x
		if not self.noReLU:
			out = self.relu(out)
		return out


class Mixed_6a(nn.Module):

	def __init__(self):
		super().__init__()

		self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)

		self.branch1 = nn.Sequential(
			BasicConv2d(256, 192, kernel_size=1, stride=1),
			BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
			BasicConv2d(192, 256, kernel_size=3, stride=2)
		)

		self.branch2 = nn.MaxPool2d(3, stride=2)

	def forward(self, x):
		x0 = self.branch0(x)
		x1 = self.branch1(x)
		x2 = self.branch2(x)
		out = torch.cat((x0, x1, x2), 1)
		return out


class Mixed_7a(nn.Module):

	def __init__(self):
		super().__init__()

		self.branch0 = nn.Sequential(
			BasicConv2d(896, 256, kernel_size=1, stride=1),
			BasicConv2d(256, 384, kernel_size=3, stride=2)
		)

		self.branch1 = nn.Sequential(
			BasicConv2d(896, 256, kernel_size=1, stride=1),
			BasicConv2d(256, 256, kernel_size=3, stride=2)
		)

		self.branch2 = nn.Sequential(
			BasicConv2d(896, 256, kernel_size=1, stride=1),
			BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
			BasicConv2d(256, 256, kernel_size=3, stride=2)
		)

		self.branch3 = nn.MaxPool2d(3, stride=2)

	def forward(self, x):
		x0 = self.branch0(x)
		x1 = self.branch1(x)
		x2 = self.branch2(x)
		x3 = self.branch3(x)
		out = torch.cat((x0, x1, x2, x3), 1)
		return out


class InceptionResnetV1(nn.Module):
	"""Inception Resnet V1 model with optional loading of pretrained weights.
	Model parameters can be loaded based on pretraining on the VGGFace2 or CASIA-Webface
	datasets. Pretrained state_dicts are automatically downloaded on model instantiation if
	requested and cached in the torch cache. Subsequent instantiations use the cache rather than
	redownloading.
	
	Keyword Arguments:
		pretrained {str} -- Optional pretraining dataset. Either 'vggface2' or 'casia-webface'.
			(default: {None})
		classify {bool} -- Whether the model should output classification probabilities or feature
			embeddings. (default: {False})
		num_classes {int} -- Number of output classes. Ignored if 'pretrained' is set, in which
			case the number of classes is set to that used for training. (default: {1001})
	"""
	def __init__(self, pretrained=None, classify=False, num_classes=1001, use_prewhiten=True):
		super().__init__()

		# Set simple attributes
		self.pretrained = pretrained
		self.classify = classify
		self.num_classes = num_classes
		self.use_prewhiten = use_prewhiten

		if pretrained == 'vggface2':
			self.num_classes = 8631
		elif pretrained == 'casia-webface':
			self.num_classes = 10575
		
		# Define layers
		self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
		self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
		self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
		self.maxpool_3a = nn.MaxPool2d(3, stride=2)
		self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
		self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
		self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)
		self.repeat_1 = nn.Sequential(
			Block35(scale=0.17),
			Block35(scale=0.17),
			Block35(scale=0.17),
			Block35(scale=0.17),
			Block35(scale=0.17),
		)
		self.mixed_6a = Mixed_6a()
		self.repeat_2 = nn.Sequential(
			Block17(scale=0.10),
			Block17(scale=0.10),
			Block17(scale=0.10),
			Block17(scale=0.10),
			Block17(scale=0.10),
			Block17(scale=0.10),
			Block17(scale=0.10),
			Block17(scale=0.10),
			Block17(scale=0.10),
			Block17(scale=0.10),
		)
		self.mixed_7a = Mixed_7a()
		self.repeat_3 = nn.Sequential(
			Block8(scale=0.20),
			Block8(scale=0.20),
			Block8(scale=0.20),
			Block8(scale=0.20),
			Block8(scale=0.20),
		)
		self.block8 = Block8(noReLU=True)
		self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
		self.last_linear = nn.Linear(1792, 512, bias=False)
		self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)

		self.logits = nn.Linear(512, self.num_classes)
		self.softmax = nn.Softmax(dim=1)


	def forward(self, x):
		"""Calculate embeddings or probabilities given a batch of input image tensors.
		
		Arguments:
			x {torch.tensor} -- Batch of image tensors representing faces.
		
		Returns:
			torch.tensor -- Batch of embeddings or softmax probabilities.
		"""
		if self.use_prewhiten:
			x = prewhiten(x)
		else:
			x = (x - 127.5) / 128
		x = self.conv2d_1a(x)
		x = self.conv2d_2a(x)
		x = self.conv2d_2b(x)
		x = self.maxpool_3a(x)
		x = self.conv2d_3b(x)
		x = self.conv2d_4a(x)
		x = self.conv2d_4b(x)
		x = self.repeat_1(x)
		x = self.mixed_6a(x)
		x = self.repeat_2(x)
		x = self.mixed_7a(x)
		x = self.repeat_3(x)
		x = self.block8(x)
		x = self.avgpool_1a(x)
		x = self.last_linear(x.view(x.shape[0], -1))
		x = self.last_bn(x)
		x = F.normalize(x, p=2, dim=1)
		if self.classify:
			x = self.logits(x)
			x = self.softmax(x)
		return x

class FaceNet(FaceModel):
	def __init__(self, dataset, use_prewhiten=True, **kwargs):
		net = InceptionResnetV1(pretrained=dataset, use_prewhiten=use_prewhiten)
		if dataset == 'vggface2':
			url = 'http://ml.cs.tsinghua.edu.cn/~dingcheng/ckpts/facenet-VGGFace2.pth'
		else:
			url = 'http://ml.cs.tsinghua.edu.cn/~dingcheng/ckpts/facenet-CASIA-Webface.pth'
		channel = 'rgb'
		FaceModel.__init__(
			self,
			net=net,
			url=url,
			channel=channel,
			**kwargs)

if __name__ == '__main__':
	img = torch.randn(1, 3, 160, 160).cuda()
	model = FaceNet(dataset='vggface2')
	out = model.forward(img)
	print(out.shape)
	model = FaceNet(dataset='casia-webface')
	out = model.forward(img)
	print(out.shape)
