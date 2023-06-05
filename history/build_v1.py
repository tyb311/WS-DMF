
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

from utils.sample import *
try:
	from attention import *

	from hrnet import *
	from dou import *
	from dmf import *
	from lunet import *
	from siam import *
except:
	from .attention import *

	from .hrnet import *
	from .dou import *
	from .dmf import *
	from .lunet import *
	from .siam import *


import configparser
def read_config(ini_file):
	''' Performs read config file and parses it.
	:param ini_file: (String) the path of a .ini file.
	:return config: (dict) the dictionary of information in ini_file.
	'''
	def _build_dict(items):
		return {item[0]: eval(item[1]) for item in items}
	# create configparser object
	cf = configparser.ConfigParser()
	# read .ini file
	cf.read(ini_file)
	config = {sec: _build_dict(cf.items(sec)) for sec in cf.sections()}
	return config


#start#
def build_model(model_type='hrdo', arch_type=None, loss_type='nce'):

	if model_type == 'hrdo':
		model = hrdo()
	elif model_type == 'dou':
		model = DoU()
	elif model_type == 'dmf':
		model = dmf32()
	elif model_type.endswith('unet'):
		model = eval(model_type+'()')
	elif model_type == 'scls':
		model = SCLSNet()
	else:
		model = eval(model_type+'()')
		# raise NotImplementedError(f'--> Unknown model_type: {model_type}')

	if arch_type == 'byol':
		model = BYOL(encoder=model, clloss=loss_type)
	elif arch_type == 'siam':
		model = SIAM(encoder=model, clloss=loss_type)
	else:
		print('No this Arch!:', arch_type)

	return model

class SCLSNet(nn.Module):#Supervised contrastive learning segmentation network
	__name__ = 'scls'
	use_render = False
	def __init__(self, cfg=None):
		super(SCLSNet, self).__init__()

		# self.fcn = build_model(cfg['net']['fcn'])
		# self.seg = build_model(cfg['net']['seg'])

		# self.projector = ProjectionMLP(16, 128, 64)
		# self.predictor = ProjectionMLP(64, 32, layers[0])
		self.fcn = lunet()#dmf32()#
		self.seg = lunet(inp_c=32)
		self.projector = ProjectionMLP(32, 128, 64)
		self.__name__ = '{}X{}'.format(self.fcn.__name__, self.seg.__name__)

	def constraint(self, **args):
		return self.fcn.constraint(**args)
	
	def regular(self, sampler, lab, fov=None, return_loss=True):
		emb = sampler.select(self.feat, self.pred.detach(), lab, fov)
		# print(emb.shape)
		emb = self.projector(emb)
		# print(emb.shape)
		self.emb = emb
		if return_loss:
			return sampler.infonce(emb)

	def forward(self, x):
		aux = self.fcn(x)
		# self.feat = F.normalize(feat, p=2, dim=1)
		self.feat = self.fcn.feat
		out = self.seg(self.feat)
		self.pred = out

		if isinstance(aux, (tuple, list)) and self.training:
			return [self.pred, aux[0], aux[1]]
		return self.pred
#end#


if __name__ == '__main__':
	x = torch.rand(2,1,64,64)

	cfg = read_config('configs/siam_unet_unet.ini')
	print(cfg)
	
	net = SCLSNet(cfg)
	net= build_model('scls', 'siam')
	# net.eval()
	ys = net(x)
	print(net.__name__)
	for y in ys:
		print(y.shape, y.min().item(), y.max().item())

	# net.train()

	sampler = MLPSampler(top=4, low=0, mode='hard')
	# net.train()
	l = net.regular(sampler, torch.rand(2,1,64,64), torch.rand(2,1,64,64), return_loss=False)


	# plot(net.emb)
	print('Params model:',sum(p.numel() for p in net.parameters() if p.requires_grad))


