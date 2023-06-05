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
def swish(x):
	# return x * torch.sigmoid(x)   #计算复杂
	return x * F.relu6(x+3)/6       #计算简单

def conv1x1(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ConvBlock(torch.nn.Module):
	attention=None
	def __init__(self, inp_c, out_c, ksize=3, shortcut=False, pool=True):
		super(ConvBlock, self).__init__()
		self.shortcut = nn.Sequential(conv1x1(inp_c, out_c), nn.BatchNorm2d(out_c))
		pad = (ksize - 1) // 2

		if pool: self.pool = nn.MaxPool2d(kernel_size=2)
		else: self.pool = False

		block = []
		block.append(nn.Conv2d(inp_c, out_c, kernel_size=ksize, padding=pad))
		block.append(nn.BatchNorm2d(out_c))
		block.append(nn.LeakyReLU())

		block.append(nn.Dropout2d(p=0.15))

		block.append(nn.Conv2d(out_c, out_c, kernel_size=ksize, padding=pad))
		block.append(nn.BatchNorm2d(out_c))
		block.append(nn.LeakyReLU())
		if self.attention=='ppolar':
			# print('ppolar')
			block.append(ParallelPolarizedSelfAttention(out_c))
		elif self.attention=='spolar':
			# print('spolar')
			block.append(SequentialPolarizedSelfAttention(out_c))
		elif self.attention=='siamam':
			# print('siamam')
			block.append(simam_module(out_c))
		# else:
		# 	print(self.attention)
		self.block = nn.Sequential(*block)
	def forward(self, x):
		if self.pool: x = self.pool(x)
		out = self.block(x)
		return swish(out + self.shortcut(x))

class UpsampleBlock(torch.nn.Module):
	def __init__(self, inp_c, out_c, up_mode='transp_conv'):
		super(UpsampleBlock, self).__init__()
		block = []
		if up_mode == 'transp_conv':
			block.append(nn.ConvTranspose2d(inp_c, out_c, kernel_size=2, stride=2))
		elif up_mode == 'up_conv':
			block.append(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False))
			block.append(nn.Conv2d(inp_c, out_c, kernel_size=1))
		else:
			raise Exception('Upsampling mode not supported')

		self.block = nn.Sequential(*block)

	def forward(self, x):
		out = self.block(x)
		return out

class ConvBridgeBlock(torch.nn.Module):
	def __init__(self, out_c, ksize=3):
		super(ConvBridgeBlock, self).__init__()
		pad = (ksize - 1) // 2
		block=[]

		block.append(nn.Conv2d(out_c, out_c, kernel_size=ksize, padding=pad))
		block.append(nn.LeakyReLU())
		block.append(nn.BatchNorm2d(out_c))

		self.block = nn.Sequential(*block)

	def forward(self, x):
		out = self.block(x)
		return out

class UpConvBlock(torch.nn.Module):
	def __init__(self, inp_c, out_c, ksize=3, up_mode='up_conv', conv_bridge=False, shortcut=False):
		super(UpConvBlock, self).__init__()
		self.conv_bridge = conv_bridge

		self.up_layer = UpsampleBlock(inp_c, out_c, up_mode=up_mode)
		self.conv_layer = ConvBlock(2 * out_c, out_c, ksize=ksize, shortcut=shortcut, pool=False)
		if self.conv_bridge:
			self.conv_bridge_layer = ConvBridgeBlock(out_c, ksize=ksize)

	def forward(self, x, skip):
		up = self.up_layer(x)
		if self.conv_bridge:
			out = torch.cat([up, self.conv_bridge_layer(skip)], dim=1)
		else:
			out = torch.cat([up, skip], dim=1)
		out = self.conv_layer(out)
		return out

class LUNet(nn.Module):
	__name__ = 'lunet'
	use_render = False
	def __init__(self, inp_c=1, n_classes=1, layers=(32,32,32,32,32), num_emb=128):
		super(LUNet, self).__init__()
		self.num_features = layers[-1]

		self.__name__ = 'u{}x{}'.format(len(layers), layers[0])
		self.n_classes = n_classes
		self.first = ConvBlock(inp_c=inp_c, out_c=layers[0], pool=False)

		self.down_path = nn.ModuleList()
		for i in range(len(layers) - 1):
			block = ConvBlock(inp_c=layers[i], out_c=layers[i + 1], pool=True)
			self.down_path.append(block)

		self.up_path = nn.ModuleList()
		reversed_layers = list(reversed(layers))
		for i in range(len(layers) - 1):
			block = UpConvBlock(inp_c=reversed_layers[i], out_c=reversed_layers[i + 1])
			self.up_path.append(block)

		# self.projector = MlpNorm(layers[0], 64, num_emb)
		# self.predictor = MlpNorm(num_emb, 64, num_emb, 2)

		self.conv_bn = nn.Sequential(
			nn.Conv2d(layers[0], layers[0], kernel_size=1),
			# nn.Conv2d(n_classes, n_classes, kernel_size=1),
			nn.BatchNorm2d(layers[0]),
		)
		self.aux = nn.Sequential(
			nn.Conv2d(layers[0], n_classes, kernel_size=1),
			# nn.Conv2d(n_classes, n_classes, kernel_size=1),
			nn.BatchNorm2d(n_classes),
			nn.Sigmoid()
		)

	# def regular(self, sampler, lab, fov=None, return_loss=True):
	# 	emb = sampler.select(self.feat, self.pred.detach(), lab, fov)
	# 	# print(emb.shape)
	# 	emb = self.projector(emb)
	# 	# print(emb.shape)
	# 	self.emb = emb
	# 	if return_loss:
	# 		return sampler.infonce(emb)

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
		self.feat = x
		# B,C,H,W = x.shape
		# x = self.projector(x.permute(0,2,3,1).reshape(-1, C))
		# x = self.predictor(x).reshape(B,-1,C).permute(0,1,2).view(B, -1, H, W)

		# if self.use_render:
		# 	aux = self.aux(self.feats)
		# 	rend = self.projector.render(self.feats, aux)
		# 	out = self.out(torch.cat([aux, rend], dim=1))
		# 	return out, aux
		self.pred = self.aux(x)
		return self.pred

def lunet(**args):
	ConvBlock.attention = None
	net = LUNet(**args)
	net.__name__ = 'lunet'
	return net

def munet(**args):
	ConvBlock.attention = 'siamam'
	net = LUNet(**args)
	net.__name__ = 'munet'
	return net
	
def punet(**args):
	ConvBlock.attention='ppolar'
	net = LUNet(**args)
	net.__name__ = 'punet'
	return net
	
def sunet(**args):
	ConvBlock.attention='spolar'
	net = LUNet(**args)
	net.__name__ = 'sunet'
	return net
#end#


class hbnet(torch.nn.Module):	#非锐化掩蔽和高提升滤波high-boost
	__name__ = 'hbnet'
	def __init__(self, n_classes=1, inp_c=1, layers=(16,16,16,16)):
		super(hbnet, self).__init__()
		self.unet1 = LUNet(inp_c, n_classes=n_classes, layers=layers)
		self.unet2 = LUNet(inp_c, n_classes=n_classes, layers=layers)
		self.my_thresh = nn.Parameter(torch.randn(size=(1,)), requires_grad=True)
		self.regular = self.unet2.regular
	tmp = {}
	def forward(self, x):
		x1 = self.unet1(x)#产生模糊图像，生成背景x1
		# x2 = torch.clamp_max(x/(x1+0.001), thresh)
		thresh = torch.abs(self.my_thresh) + .6
		x2 = torch.clamp_min(x1, 0.001)
		x2 = torch.max(x/(x1+0.001), thresh)
		# print(x2.shape)
		x2 = self.unet2(x2)
		
		self.tmp['x1'] = x1
		self.tmp['x2'] = x2
		self.feat = self.unet2.feat
		self.pred = x2
		if self.training:
			return [x2,x1]
		return x2


if __name__ == '__main__':
	import time

	net = lunet()
	net = munet()
	# net = punet()
	# net = sunet()
	# net = hbnet()


	x = torch.rand(2,1,64,64)

	st = time.time()
	ys = net(x)
	print('time:', time.time()-st)

	for y in ys:
		print('pred:', y.shape)
	# print(net.__name__, y['loss'])

	sampler = MLPSampler(top=4, low=0, mode='hard')
	# net.train()
	# l = net.regular(sampler, torch.rand(2,1,64,64), torch.rand(2,1,64,64), return_loss=False)
	# print(net.__name__, l.item())

	# plot(net.emb)
	print('Params model:',sum(p.numel() for p in net.parameters() if p.requires_grad))