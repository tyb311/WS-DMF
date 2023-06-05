
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
class MlpSphere(nn.Module):#球极平面射影
	def __init__(self, dim_inp=64, dim_out=32):
		super(MlpSphere, self).__init__()
		# dim_mid = max(dim_inp, dim_out)//2
		# hidden layers
		linear_hidden = []#nn.Identity()
		# for i in range(num_layers - 1):
		linear_hidden.append(nn.Linear(dim_inp, dim_out))
		linear_hidden.append(nn.BatchNorm1d(dim_out))
		linear_hidden.append(nn.Dropout(p=0.2))
		linear_hidden.append(nn.LeakyReLU())
		self.linear_hidden = nn.Sequential(*linear_hidden)

	def forward(self, x):
		x = self.linear_hidden(x)
		return x

class MlpNorm(nn.Module):
	def __init__(self, dim_inp=256, dim_out=64):
		super(MlpNorm, self).__init__()
		dim_mid = min(dim_inp, dim_out)#max(dim_inp, dim_out)//2
		# hidden layers
		linear_hidden = []#nn.Identity()
		# for i in range(num_layers - 1):
		linear_hidden.append(nn.Linear(dim_inp, dim_mid))
		linear_hidden.append(nn.BatchNorm1d(dim_mid))
		linear_hidden.append(nn.LeakyReLU())
		self.linear_hidden = nn.Sequential(*linear_hidden)

		self.linear_out = nn.Linear(dim_mid, dim_out)# if num_layers >= 1 else nn.Identity()

	def forward(self, x):
		x = self.linear_hidden(x)
		x = self.linear_out(x)
		return F.normalize(x, p=2, dim=-1)

def torch_dilation(x, ksize=3, stride=1):
	return F.max_pool2d(x, (ksize, ksize), stride, ksize//2)

class MorphBlock(nn.Module):
    def __init__(self, channel=8):
        super().__init__()
        self.ch_wv = nn.Sequential(
            nn.Conv2d(2,channel,kernel_size=5, padding=2),
            nn.Conv2d(channel,channel,kernel_size=5, padding=2),
			nn.BatchNorm2d(channel),
            nn.Conv2d(channel,channel//2,kernel_size=3, padding=1),
        )
        self.ch_wq = nn.Sequential(
            nn.Conv2d(channel//2,8,kernel_size=3, padding=1),
			nn.BatchNorm2d(8),
            nn.Conv2d(8,1,kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = torch.cat([torch_dilation(x, ksize=3), x], dim=1)#, 1-torch_dilation(1-x, ksize=3)
        return self.ch_wq(self.ch_wv(x))

class SeqNet(nn.Module):#Supervised contrastive learning segmentation network
	__name__ = 'scls'
	def __init__(self, type_net, type_seg, num_emb=128):
		super(SeqNet, self).__init__()

		self.fcn = eval(type_net+'(num_emb=num_emb)')#build_model(cfg['net']['fcn'])
		self.seg = eval(type_seg+'(inp_c=32)')#build_model(cfg['net']['seg'])

		self.projector = MlpNorm(32, num_emb)#self.fcn.projector#MlpNorm(32, 64, num_emb)
		self.predictor = MlpNorm(num_emb, num_emb)#self.fcn.predictor#MlpNorm(32, 64, num_emb)

		self.morpholer1 = MorphBlock()#形态学模块使用一个还是两个哪？
		self.morpholer2 = MorphBlock()#形态学模块使用一个还是两个哪？
		self.__name__ = '{}X{}'.format(self.fcn.__name__, self.seg.__name__)

	def constraint(self, aux=None, fun=None, **args):
		aux = torch_dilation(aux)
		los1 = fun(self.sdm1, aux)
		los2 = fun(self.sdm2, aux)
		if self.__name__.__contains__('dmf'):
			los1 = los1 + self.fcn.regular()*0.1
		return los1 + los2
	
	def regular(self, sampler, lab, fov=None, return_loss=True):
		emb = sampler.select(self.feat.clone(), self.pred.detach(), lab, fov)
		# print(emb.shape)
		emb = self.projector(emb)
		# print(emb.shape)
		self.emb = emb
		if return_loss:
			return sampler.infonce(emb)
	tmp = {}
	def forward(self, x):
		aux = self.fcn(x)
		self.feat = self.fcn.feat
		out = self.seg(self.feat)
		self.pred = out

		self.sdm1 = self.morpholer1(aux)
		self.sdm2 = self.morpholer2(out)
		self.tmp = {'sdm1':self.sdm1, 'sdm2':self.sdm2}

		if self.training:
			if isinstance(aux, (tuple, list)):
				return [self.pred, aux[0], aux[1]]
			else:
				return [self.pred, aux]
		return self.pred

class SphereUNet(nn.Module):
	__name__ = 'sphu'
	def __init__(self, inp_c=1, n_classes=1, layers=(32,32,32,32,32), num_emb=32, use_sim=True):
		super(SphereUNet, self).__init__()
		self.__name__ = 'spu' + ('Sim' if use_sim else 'Nce')
		self.use_sim=use_sim
		self.num_features = layers[-1]
		self.n_classes = n_classes
		self.first = BasicConv2d(inp_c, layers[0])

		self.down_path = nn.ModuleList()
		for i in range(len(layers) - 1):
			block = ConvBlock(inp_c=layers[i], out_c=layers[i + 1], pool=True)
			self.down_path.append(block)

		self.up_path = nn.ModuleList()
		reversed_layers = list(reversed(layers))
		for i in range(len(layers) - 1):
			block = UpConvBlock(inp_c=reversed_layers[i], out_c=reversed_layers[i + 1])
			self.up_path.append(block)

		self.projector = MlpNorm(layers[0], num_emb)
		self.spherepro = MlpSphere(num_emb, 32)
		self.out = OutSigmoid(32, 1)
		self.morpholer = MorphBlock()#形态学模块使用一个还是两个哪？

		self.conv_bn = nn.Sequential(
			nn.Conv2d(layers[0], layers[0], kernel_size=1),
			# nn.Conv2d(n_classes, n_classes, kernel_size=1),
			nn.BatchNorm2d(layers[0]),
		)

	def constraint(self, aux=None, fun=None, **args):
		aux = torch_dilation(aux)
		los = fun(self.sdm, aux)
		return los

	def regular(self, sampler, lab, fov=None, *args):
		feat = sampler.select(self.feat, self.pred.detach(), lab, fov)
		self.feat = feat
		if self.use_sim:
			return similar_matrix2(feat, feat.detach())
		return infonce_matrix2(feat, feat.detach())

	def forward(self, x):
		x = self.first(x)
		down_activations = []
		for i, down in enumerate(self.down_path):
			down_activations.append(x)
			# print(x.shape)
			x = down(x)
		down_activations.reverse()

		for i, up in enumerate(self.up_path):
			x = up(x, down_activations[i])

		# self.feat = F.normalize(x, dim=1, p=2)
		x = self.conv_bn(x)

		B,C,H,W = x.shape
		x = self.projector(x.permute(0,2,3,1).reshape(-1, C))
		self.proj = x.clone()#.reshape(-1, C)
		self.feat = self.proj.reshape(B,-1,self.proj.shape[-1]).permute(0,2,1).reshape(B, -1, H, W)

		out = self.spherepro(self.proj)
		# print('proj & pred:', self.proj.shape, self.pred.shape, out.shape)#torch.Size([2, 128, 64, 64])
		out = out.reshape(B,-1,out.shape[-1]).permute(0,2,1).reshape(B, -1, H, W)
		# print('sphere proj:', out.shape)#torch.Size([2, 128, 64, 64])
		self.pred = self.out(out)

		self.sdm = self.morpholer(self.pred)
		self.tmp = {'sdm':self.sdm}
		return self.pred
def spus(**args):
	return SphereUNet(use_sim=True, **args)
def spun(**args):
	return SphereUNet(use_sim=False, **args)

def build_model(type_net='hrdo', type_seg='', type_loss='sim2', type_arch='', num_emb=128):

	if type_net == 'hrdo':
		model = hrdo()
	elif type_net == 'dou':
		model = DoU()
	elif type_net == 'dmf':
		model = dmf32()
	else:
		model = eval(type_net+'(num_emb=num_emb)')
		# raise NotImplementedError(f'--> Unknown type_net: {type_net}')

	if type_seg!='':
		model = SeqNet(type_net, type_seg, num_emb=num_emb)
		if type_arch=='siam':
			model = SIAM(encoder=model, clloss=type_loss, proj_num_length=num_emb)
		elif type_arch=='roma':
			model = ROMA(encoder=model, clloss=type_loss, proj_num_length=num_emb)

	return model
#end#

import time
if __name__ == '__main__':
	num_emb = 128
	x = torch.rand(8,1,128,128)

	# cfg = read_config('configs/siam_unet_unet.ini')
	# print(cfg)
	
	# net = build_model('sunet', 'munet', 'sim2', 'roma', num_emb)
	# net = build_model('sunet', 'munet', 'sim2', 'siam', num_emb)
	# net = build_model('dmf32', 'munet', 'sim2', '', num_emb)
	# net = build_model('sunet', 'munet', 'arc', 'siam', num_emb)
	# net = build_model('sunet', 'sunet', '', '', num_emb)
	net = build_model('spun', '', '', '', num_emb)
	# net.eval()

	st = time.time()
	ys = net(x)
	print(net.__name__, 'Time:', time.time() - st)
	for y in ys:
		print(y.shape, y.min().item(), y.max().item())

	# net.train()
	for key, item in net.tmp.items():
		print(key, item.shape)

	sampler = MLPSampler(top=4, low=0, mode='half')
	# net.train()
	st = time.time()
	l = net.regular(sampler, torch.rand_like(x), torch.rand_like(x))
	print('Regular:', l.item())
	print(net.__name__, 'Time:', time.time() - st)
	plot4(emb=net.feat, path_save='emb.png')

	
	st = time.time()
	l = net.constraint(aux=x, fun=nn.MSELoss())
	print('constraint:', l.item())
	print(net.__name__, 'Time:', time.time() - st)

	# plot(net.emb)
	print('Params model:',sum(p.numel() for p in net.parameters() if p.requires_grad))


