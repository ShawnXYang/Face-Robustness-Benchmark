import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable

from RobFR.networks.FaceModel import FaceModel

class MobileNet(nn.Module):
	def __init__(self, stride):
		super(MobileNet, self).__init__()

		def conv_bn(inp, oup, stride):
			return nn.Sequential(
				nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
				nn.BatchNorm2d(oup),
				nn.ReLU(inplace=True)
			)

		def conv_dw(inp, oup, stride):
			return nn.Sequential(
				nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
				nn.BatchNorm2d(inp),
				nn.ReLU(inplace=True),
	
				nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
				nn.BatchNorm2d(oup),
				nn.ReLU(inplace=True),
			)

		self.model = nn.Sequential(
			conv_bn(  3,  32, stride), 
			conv_dw( 32,  64, 1),
			conv_dw( 64, 128, 2),
			conv_dw(128, 128, 1),
			conv_dw(128, 256, 2),
			conv_dw(256, 256, 1),
			conv_dw(256, 512, 2),
			conv_dw(512, 512, 1),
			conv_dw(512, 512, 1),
			conv_dw(512, 512, 1),
			conv_dw(512, 512, 1),
			conv_dw(512, 512, 1),
			conv_dw(512, 1024, 2),
			conv_dw(1024, 1024, 1),
			#nn.AvgPool2d(7),
		)
		if stride == 2:
			self.linear7 = nn.Conv2d(1024, 1024, 4, 1, 0, groups=1024, bias=False) 
		elif stride == 1:
			self.linear7 = nn.Conv2d(1024, 1024, 7, 1, 0, groups=1024, bias=False) 
		else:
			NotImplementedError
		self.fc = nn.Linear(1024, 512)

	def forward(self, x):
		#print(x.size())
		x = (x - 127.5) / 128
		x = self.model(x)
		x = self.linear7(x)
		#print(x.size())
		x = x.view(-1, 1024)
		x = self.fc(x)
		return x


class Mobilenet(FaceModel):
	def __init__(self, stride=2, **kwargs):
		Net = MobileNet(stride=stride)
		if stride == 2:
			url = 'http://ml.cs.tsinghua.edu.cn/~dingcheng/ckpts/face_models/model11/Backbone_Mobilenet_Epoch_125_Batch_710750_Time_2019-04-14-18-15_checkpoint.pth'
		else:
			url = 'http://ml.cs.tsinghua.edu.cn/~zihao/realai/ckpts/model27/Backbone_Mobilenet_Epoch_124_Batch_1410128_Time_2019-04-30-14-43_checkpoint.pth'
		channel = 'bgr'
		FaceModel.__init__(
			self,
			net=Net,
			url=url,
			channel=channel,
			**kwargs)

if __name__ == "__main__":
	model = Mobilenet()
