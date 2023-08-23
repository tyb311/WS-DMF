
# 常用资源库
import pandas as pd
import numpy as np
EPS = 1e-9#
import os,glob,numbers
# 图像处理
import math,cv2,random
from PIL import Image, ImageFile, ImageOps, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True
# 图像显示
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as f
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader



import sys
sys.path.append('.')
sys.path.append('..')

from utils import *
from nets import *
from scls import *



#start#

class SeqNet(nn.Module):#Supervised contrastive learning segmentation network
	__name__ = 'scls'
	def __init__(self, type_net, type_seg, num_emb=128):
		super(SeqNet, self).__init__()

		self.fcn = eval(type_net+'(num_emb=num_emb)')#build_model(cfg['net']['fcn'])
		self.seg = eval(type_seg+'(inp_c=32)')#build_model(cfg['net']['seg'])

		self.projector = MlpNorm(32, num_emb)#self.fcn.projector#MlpNorm(32, 64, num_emb)
		self.predictor = MlpNorm(num_emb, num_emb)#self.fcn.predictor#MlpNorm(32, 64, num_emb)

		self.morpholer1 = MorphBlock(32+2)#形态学模块使用一个还是两个哪？
		self.morpholer2 = MorphBlock(32+2)#形态学模块使用一个还是两个哪？
		self.__name__ = '{}X{}'.format(self.fcn.__name__, self.seg.__name__)

	def constraint(self, aux=None, fun=None, **args):
		aux = torch_dilation(aux)
		los1 = fun(self.sdm1, aux)
		los2 = fun(self.sdm2, aux)
		# if self.__name__.__contains__('dmf'):
		# 	los1 = los1 + self.fcn.regular()*0.1
		return los1, los2
	
	tmp = {}
	def forward(self, x):
		aux = self.fcn(x)
		self.feat = self.fcn.feat
		out = self.seg(self.feat)
		self.pred = out
		# print(self.fcn.feat.shape, self.seg.feat.shape)
		self.sdm1 = self.morpholer1(self.fcn.feat, aux)
		self.sdm2 = self.morpholer2(self.seg.feat, out)
		self.tmp = {'sdm1':self.sdm1, 'sdm2':self.sdm2}

		if self.training:
			if isinstance(aux, (tuple, list)):
				return [self.pred, aux[0], aux[1]]
			else:
				return [self.pred, aux]
		return self.pred


def build_model(type_net='hrdo', type_seg='', type_loss='sim2', type_arch='', num_emb=128):

	model = eval(type_net+'(num_emb=num_emb)')
		# raise NotImplementedError(f'--> Unknown type_net: {type_net}')

	if type_seg!='':
		model = SeqNet(type_net, type_seg, num_emb=num_emb)

	return model
#end#


#	把形态学模块放到前一层
import time
if __name__ == '__main__':
	num_emb = 128
	x = torch.rand(8,1,128,128)

	# cfg = read_config('configs/siam_unet_unet.ini')
	# print(cfg)
	
	# net = build_model('sunet', 'munet', 'sim2', 'roma', num_emb)
	# net = build_model('lunet', 'munet', 'sim2', 'siam', num_emb)
	# net = build_model('smf', 'lunet', 'sim2', 'siam', num_emb)
	net = build_model('smf', 'lunet', 'sim2', '', num_emb)
	# net = build_model('dmf32', 'munet', 'sim2', '', num_emb)
	# net = build_model('sunet', 'munet', 'arc', 'siam', num_emb)
	# net = build_model('lunet', 'lunet', '', '', num_emb)
	# net = build_model('spun', '', '', '', num_emb)
	# net.eval()

	st = time.time()
	ys = net(x)
	print(net.__name__, 'Time:', time.time() - st)
	for y in ys:
		print(y.shape, y.min().item(), y.max().item())

	# # net.train()
	# for key, item in net.tmp.items():
	# 	print(key, item.shape)

	# sampler = MLPSampler(top=4, low=0, mode='half')
	# # net.train()
	# st = time.time()
	# l = net.regular(sampler, torch.rand_like(x), torch.rand_like(x))
	# # print('Regular:', l.item())
	# print(net.__name__, 'Time:', time.time() - st)
	# print('feat:', net.feat.shape, net.proj.shape)
	# # plot4(emb=net.feat, path_save='emb.png')
	# # plt.show()

	
	# st = time.time()
	# l = net.constraint(aux=x, fun=nn.MSELoss())
	# print('constraint:', l.item())
	# print(net.__name__, 'Time:', time.time() - st)

	# plot(net.emb)
	print('Params model:',sum(p.numel() for p in net.parameters() if p.requires_grad))


