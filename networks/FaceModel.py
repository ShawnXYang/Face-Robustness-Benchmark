import os
from six.moves import urllib
import torch

def get_model(url, net, embedding_size, no_gpu=False):
	"""
		:param url: a string, the url
		:param net: the backbone model
		:param embedding_size: int, output feature shape
		:param no_gpu: boolean, whether to use gpu
	"""

	model_name = url.split('/')[-1]
	try:
		print('Load existing checkpoint')
		checkpoint = torch.load('./ckpts/{}'.format(model_name), 
				map_location=lambda storage, loc: storage.cuda())
	except Exception:
		print('No existing checkpoint, now downloading online')
		if not os.path.exists('./ckpts/'):
			try:
				os.makedirs('./ckpts/')
			except OSError as e:
				if e.errno != errno.EEXIST:
					raise
		urllib.request.urlretrieve(
			url,
			'./ckpts/{}'.format(model_name))
		print('Finish downloading')
		print('Load checkpoint')
		checkpoint = torch.load('./ckpts/{}'.format(model_name), 
				map_location=lambda storage, loc: storage.cuda())

	if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
		net.load_state_dict(checkpoint['state_dict'])
	else:
		net.load_state_dict(checkpoint)

	net.eval()
	output = {'prelogits': (net, embedding_size)}
	if not no_gpu:
		for key, (graph, _) in output.items():
			graph = graph.cuda()
		input_device = torch.device("cuda")
	else:
		input_device = torch.device("cpu")
	return output, input_device

class FaceModel:
	def __init__(self, url, net, **kwargs):
		embedding_size = kwargs.get('embedding_size', 512)
		no_gpu = kwargs.get('no_gpu', False)
		# get the pytorch model
		output, input_device = get_model(
				net=net,
				url=url,
				embedding_size=embedding_size,
				no_gpu=no_gpu)
		out_dims = embedding_size
		self.out_dims = out_dims
		self.channel = kwargs.get('channel', 'rgb')
		self.output = output
		self.input_device = input_device
	def forward(self, x, use_prelogits=False):
		if self.channel == 'bgr':
			x = torch.flip(x, dims=[1])
		x = self.output['prelogits'][0](x)
		if not use_prelogits:
			x = x / torch.sqrt(torch.sum(x**2, dim=1, keepdim=True) + 1e-5)
		return x
	def zero_grad(self):
		self.output['prelogits'][0].zero_grad()
