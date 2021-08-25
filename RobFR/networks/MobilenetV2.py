import torch.nn as nn
import math
import torch
from torch.autograd import Variable

from RobFR.networks.FaceModel import FaceModel

def conv_bn(inp, oup, stride):
	return nn.Sequential(
		nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
		nn.BatchNorm2d(oup),
		nn.ReLU6(inplace=True)
	)


def conv_1x1_bn(inp, oup):
	return nn.Sequential(
		nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
		nn.BatchNorm2d(oup),
		nn.ReLU6(inplace=True)
	)


class InvertedResidual(nn.Module):
	def __init__(self, inp, oup, stride, expand_ratio):
		super(InvertedResidual, self).__init__()
		self.stride = stride
		assert stride in [1, 2]

		hidden_dim = int(round(inp * expand_ratio))
		self.use_res_connect = self.stride == 1 and inp == oup

		if expand_ratio == 1:
			self.conv = nn.Sequential(
				# dw
				nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
				nn.BatchNorm2d(hidden_dim),
				nn.ReLU6(inplace=True),
				# pw-linear
				nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
				nn.BatchNorm2d(oup),
			)
		else:
			self.conv = nn.Sequential(
				# pw
				nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
				nn.BatchNorm2d(hidden_dim),
				nn.ReLU6(inplace=True),
				# dw
				nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
				nn.BatchNorm2d(hidden_dim),
				nn.ReLU6(inplace=True),
				# pw-linear
				nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
				nn.BatchNorm2d(oup),
			)

	def forward(self, x):
		if self.use_res_connect:
			return x + self.conv(x)
		else:
			return self.conv(x)


class MobileNetV2(nn.Module):
	def __init__(self, n_class=1000, input_size=224, width_mult=1., stride=2):
		super(MobileNetV2, self).__init__()
		block = InvertedResidual
		input_channel = 32
		last_channel = 1280
		interverted_residual_setting = [
			# t, c, n, s
			[1, 16, 1, 1],
			[6, 24, 2, 2],
			[6, 32, 3, 2],
			[6, 64, 4, 2],
			[6, 96, 3, 1],
			[6, 160, 3, 2],
			[6, 320, 1, 1],
		]

		# building first layer
		assert input_size % 32 == 0
		input_channel = int(input_channel * width_mult)
		self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
		self.features = [conv_bn(3, input_channel, stride)]
		# building inverted residual blocks
		for t, c, n, s in interverted_residual_setting:
			output_channel = int(c * width_mult)
			for i in range(n):
				if i == 0:
					self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
				else:
					self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
				input_channel = output_channel
		# building last several layers
		self.features.append(conv_1x1_bn(input_channel, self.last_channel))
		# make it nn.Sequential
		self.features = nn.Sequential(*self.features)

		if stride == 2:
			self.linear7 = nn.Conv2d(1280, 1280, 4, 1, 0, groups=1280, bias=False) 
		elif stride == 1:
			self.linear7 = nn.Conv2d(1280, 1280, 7, 1, 0, groups=1280, bias=False) 
		else:
			NotImplementedError
		# building classifier
		self.classifier = nn.Sequential(
			nn.Dropout(0.2),
			nn.Linear(1280, 512),
		)

		self._initialize_weights()

	def forward(self, x):
		x = (x - 127.5) / 128
		x = self.features(x)
		#print(x.size())
		#x = x.mean(3).mean(2)
		x = self.linear7(x)
		x = x.view(-1, 1280)
		#print(x.size())
		x = self.classifier(x)
		return x

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()

class MobilenetV2(FaceModel):
	def __init__(self, stride=2, **kwargs):#, use_all_features=False:
		Net = MobileNetV2(stride=stride)
		if stride == 2:
			url = 'http://ml.cs.tsinghua.edu.cn/~dingcheng/ckpts/face_models/model12/Backbone_Mobilenetv2_Epoch_110_Batch_625460_Time_2019-04-14-17-30_checkpoint.pth'
		elif stride == 1:
			url = 'http://ml.cs.tsinghua.edu.cn/~zihao/realai/ckpts/model28/Backbone_Mobilenetv2_Epoch_50_Batch_606500_Time_2019-04-25-08-57_checkpoint.pth'
		channel = 'bgr'
		FaceModel.__init__(
			self,
			url=url,
			net=Net,
			channel=channel,
			**kwargs)

if __name__ == "__main__":
	model = MobilenetV2(stride=1)
