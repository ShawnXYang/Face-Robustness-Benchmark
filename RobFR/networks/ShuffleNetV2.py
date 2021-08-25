import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math

from torch.autograd import Variable

from RobFR.networks.FaceModel import FaceModel

def conv_bn(inp, oup, stride):
	return nn.Sequential(
		nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
		nn.BatchNorm2d(oup),
		nn.ReLU(inplace=True)
	)


def conv_1x1_bn(inp, oup):
	return nn.Sequential(
		nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
		nn.BatchNorm2d(oup),
		nn.ReLU(inplace=True)
	)

def channel_shuffle(x, groups):
	batchsize, num_channels, height, width = x.data.size()

	channels_per_group = num_channels // groups
	
	# reshape
	x = x.view(batchsize, groups, 
		channels_per_group, height, width)

	x = torch.transpose(x, 1, 2).contiguous()

	# flatten
	x = x.view(batchsize, -1, height, width)

	return x
	
class InvertedResidual(nn.Module):
	def __init__(self, inp, oup, stride, benchmodel):
		super(InvertedResidual, self).__init__()
		self.benchmodel = benchmodel
		self.stride = stride
		assert stride in [1, 2]

		oup_inc = oup//2
		
		if self.benchmodel == 1:
			#assert inp == oup_inc
			self.banch2 = nn.Sequential(
				# pw
				nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
				nn.BatchNorm2d(oup_inc),
				nn.ReLU(inplace=True),
				# dw
				nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
				nn.BatchNorm2d(oup_inc),
				# pw-linear
				nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
				nn.BatchNorm2d(oup_inc),
				nn.ReLU(inplace=True),
			)				 
		else:				   
			self.banch1 = nn.Sequential(
				# dw
				nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
				nn.BatchNorm2d(inp),
				# pw-linear
				nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
				nn.BatchNorm2d(oup_inc),
				nn.ReLU(inplace=True),
			)		 
	
			self.banch2 = nn.Sequential(
				# pw
				nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
				nn.BatchNorm2d(oup_inc),
				nn.ReLU(inplace=True),
				# dw
				nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
				nn.BatchNorm2d(oup_inc),
				# pw-linear
				nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
				nn.BatchNorm2d(oup_inc),
				nn.ReLU(inplace=True),
			)
		  
	@staticmethod
	def _concat(x, out):
		# concatenate along channel axis
		return torch.cat((x, out), 1)		 

	def forward(self, x):
		if 1==self.benchmodel:
			x1 = x[:, :(x.shape[1]//2), :, :]
			x2 = x[:, (x.shape[1]//2):, :, :]
			out = self._concat(x1, self.banch2(x2))
		elif 2==self.benchmodel:
			out = self._concat(self.banch1(x), self.banch2(x))

		return channel_shuffle(out, 2)


class ShufflenetV2(nn.Module):
	def __init__(self, n_class=1000, input_size=224, width_mult=1., stride=2, pooling='Linear'):
		super(ShufflenetV2, self).__init__()
		
		assert input_size % 32 == 0
		
		self.stage_repeats = [4, 8, 4]
		# index 0 is invalid and should never be called.
		# only used for indexing convenience.
		if width_mult == 0.5:
			self.stage_out_channels = [-1, 24, 48,	96, 192, 1024]
		elif width_mult == 1.0:
			self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
		elif width_mult == 1.5:
			self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
		elif width_mult == 2.0:
			self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
		else:
			raise ValueError(
				"""{} groups is not supported for
					   1x1 Grouped Convolutions""".format(num_groups))

		# building first layer
		input_channel = self.stage_out_channels[1]
		self.conv1 = conv_bn(3, input_channel, stride=stride)	 
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.features = []
		# building inverted residual blocks
		for idxstage in range(len(self.stage_repeats)):
			numrepeat = self.stage_repeats[idxstage]
			output_channel = self.stage_out_channels[idxstage+2]
			for i in range(numrepeat):
				if i == 0:
				#inp, oup, stride, benchmodel):
					self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
				else:
					self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
				input_channel = output_channel
				
				
		# make it nn.Sequential
		self.features = nn.Sequential(*self.features)
		# building last several layers
		self.conv_last		= conv_1x1_bn(input_channel, self.stage_out_channels[-1])
	
		if stride == 1:
			width = 7
		elif stride == 2:
			width = 4
		else:
			raise NotImplementedError

		self.pooling = pooling
		if pooling == 'Linear':
			self.linear7 = nn.Linear(1024 * width * width, 1024)
		elif pooling == 'GDConv':
			self.linear7 = nn.Conv2d(1024, 1024, width, 1, 0, groups=1024, bias=False )
		else:
			self.linear7 = nn.AvgPool2d(width)
		# building classifier
		self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], 512))

	def forward(self, x):
		x = (x - 127.5) / 128
		x = self.conv1(x)
		x = self.maxpool(x)
		x = self.features(x)
		x = self.conv_last(x)
		#print(x.data.size())
		#x = x.data
		if self.pooling == 'Linear':
			x = x.view(x.size(0), -1)
		x = self.linear7(x)
		x = x.view(-1, self.stage_out_channels[-1])
		x = self.classifier(x)
		return x
class ShuffleNetV2(FaceModel):
	def __init__(self, stride=2, pooling='Linear', width_mult=1., **kwargs):
		net = ShufflenetV2(pooling=pooling, stride=stride, width_mult=width_mult)
		if pooling == 'Linear':
			url = 'http://ml.cs.tsinghua.edu.cn/~zihao/realai/ckpts/model14_2/Backbone_ShuffleNetV2_Epoch_11_Batch_62546_Time_2019-04-14-07-49_checkpoint.pth'
		elif pooling == 'AvgPool':
			url = 'http://ml.cs.tsinghua.edu.cn/~zihao/realai/ckpts/model14_3/Backbone_ShuffleNetV2_Epoch_24_Batch_136464_Time_2019-04-15-13-51_checkpoint.pth'
		elif pooling == 'GDConv':
			url = 'http://ml.cs.tsinghua.edu.cn/~zihao/realai/ckpts/model19_1/Backbone_ShuffleNetV2_Epoch_41_Batch_466252_Time_2019-04-23-14-31_checkpoint.pth'
		channel = 'bgr'
		FaceModel.__init__(
			self,
			url=url,
			net=net,
			channel=channel,
			**kwargs)

if __name__ == "__main__":
	model = ShuffleNetV2()
