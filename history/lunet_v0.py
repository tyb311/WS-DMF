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
from attention import *
from hrnet import *

#start#
def conv1x1(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ConvBlock(torch.nn.Module):
	attention=None
	def __init__(self, in_channels, out_c, k_sz=3, shortcut=False, pool=True):
		super(ConvBlock, self).__init__()
		self.shortcut = nn.Sequential(conv1x1(in_channels, out_c), nn.BatchNorm2d(out_c))
		pad = (k_sz - 1) // 2

		block = []
		if pool: self.pool = nn.MaxPool2d(kernel_size=2)
		else: self.pool = False

		block.append(nn.Conv2d(in_channels, out_c, kernel_size=k_sz, padding=pad))
		block.append(nn.ReLU())
		block.append(nn.BatchNorm2d(out_c))

		block.append(nn.Dropout2d(p=0.15))

		block.append(nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding=pad))
		block.append(nn.ReLU())
		block.append(nn.BatchNorm2d(out_c))
		if self.attention=='ppolar':
			# print('ppolar')
			block.append(ParallelPolarizedSelfAttention(out_c))
		elif self.attention=='spolar':
			# print('spolar')
			block.append(SequentialPolarizedSelfAttention(out_c))

		self.block = nn.Sequential(*block)
	def forward(self, x):
		if self.pool: x = self.pool(x)
		out = self.block(x)
		return out + self.shortcut(x)

class UpsampleBlock(torch.nn.Module):
	def __init__(self, in_channels, out_c, up_mode='transp_conv'):
		super(UpsampleBlock, self).__init__()
		block = []
		if up_mode == 'transp_conv':
			block.append(nn.ConvTranspose2d(in_channels, out_c, kernel_size=2, stride=2))
		elif up_mode == 'up_conv':
			block.append(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False))
			block.append(nn.Conv2d(in_channels, out_c, kernel_size=1))
		else:
			raise Exception('Upsampling mode not supported')

		self.block = nn.Sequential(*block)

	def forward(self, x):
		out = self.block(x)
		return out

class ConvBridgeBlock(torch.nn.Module):
	def __init__(self, out_c, k_sz=3):
		super(ConvBridgeBlock, self).__init__()
		pad = (k_sz - 1) // 2
		block=[]

		block.append(nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding=pad))
		block.append(nn.ReLU())
		block.append(nn.BatchNorm2d(out_c))

		self.block = nn.Sequential(*block)

	def forward(self, x):
		out = self.block(x)
		return out

class UpConvBlock(torch.nn.Module):
	def __init__(self, in_channels, out_c, k_sz=3, up_mode='up_conv', conv_bridge=False, shortcut=False):
		super(UpConvBlock, self).__init__()
		self.conv_bridge = conv_bridge

		self.up_layer = UpsampleBlock(in_channels, out_c, up_mode=up_mode)
		self.conv_layer = ConvBlock(2 * out_c, out_c, k_sz=k_sz, shortcut=shortcut, pool=False)
		if self.conv_bridge:
			self.conv_bridge_layer = ConvBridgeBlock(out_c, k_sz=k_sz)

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
	def __init__(self, in_channels=1, n_classes=1, layers=(32,32,32,32,32)):
		super(LUNet, self).__init__()
		self.num_features = layers[-1]

		self.__name__ = 'u{}x{}'.format(len(layers), layers[0])
		self.n_classes = n_classes
		self.first = ConvBlock(in_channels=in_channels, out_c=layers[0], pool=False)

		self.down_path = nn.ModuleList()
		for i in range(len(layers) - 1):
			block = ConvBlock(in_channels=layers[i], out_c=layers[i + 1], pool=True)
			self.down_path.append(block)

		self.up_path = nn.ModuleList()
		reversed_layers = list(reversed(layers))
		for i in range(len(layers) - 1):
			block = UpConvBlock(in_channels=reversed_layers[i], out_c=reversed_layers[i + 1])
			self.up_path.append(block)

		self.projection = ProjectionMLP(32)
		# self.out = nn.Sequential(
		# 	nn.Conv2d(1+self.projection.dim_rend, n_classes, kernel_size=1),
		# 	# nn.Conv2d(n_classes, n_classes, kernel_size=1),
		# 	nn.BatchNorm2d(n_classes),
		# 	out
		# )
		self.aux = nn.Sequential(
			nn.Conv2d(layers[0], n_classes, kernel_size=1),
			# nn.Conv2d(n_classes, n_classes, kernel_size=1),
			nn.BatchNorm2d(n_classes),
			nn.Sigmoid()
		)
	def regular(self, sampler, lab, fov=None):
		emb = sampler.select(self.feat, self.pred.detach(), lab, fov)
		# print(emb.shape)
		emb = self.projection(emb)
		# print(emb.shape)
		self.emb = emb
		return sampler.infonce(emb)

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

		self.feat = F.normalize(x, dim=1, p=2)
		self.feat = x

		# if self.use_render:
		# 	aux = self.aux(self.feats)
		# 	rend = self.projection.render(self.feats, aux)
		# 	out = self.out(torch.cat([aux, rend], dim=1))
		# 	return out, aux
		self.pred = self.aux(self.feat)
		return self.pred

def lunet():
	net = LUNet()
	net.__name__ = 'lunet'
	return net
	
def punet():
	ConvBlock.attention='ppolar'
	net = LUNet()
	net.__name__ = 'punet'
	return net
	
def sunet():
	ConvBlock.attention='spolar'
	net = LUNet()
	net.__name__ = 'sunet'
	return net
#end#


if __name__ == '__main__':
	net = lunet()
	net = punet()
	# net = sunet()


	sampler = MLPSampler(top=4, low=0, mode='hard', temp=0.2, ver='v3')

	x = torch.rand(2,1,64,64)

	net.eval()
	ys = net(x)

	for y in ys:
		print(y.shape)
	# print(net.__name__, y['loss'])

	# net.train()
	l = net.regular(sampler, torch.rand(2,1,64,64), torch.rand(2,1,64,64))
	print(net.__name__, l.item())


	# plot(net.emb)

	# plot(net.emb)
	print('Params model:',sum(p.numel() for p in net.parameters() if p.requires_grad))