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
	from hrnet import *
except:
	from .hrnet import *

# class MlpNorm(nn.Module):
# 	def __init__(self, d_inp=256, d_mid=128, d_out=64, d_red=8, num_layers=2):
# 		super(MlpNorm, self).__init__()
# 		self.dim_rend = d_red
# 		# hidden layers
# 		linear_hidden = [nn.Identity()]
# 		for i in range(num_layers - 1):
# 			linear_hidden.append(nn.Linear(d_inp if i == 0 else d_mid, d_mid))
# 			linear_hidden.append(nn.BatchNorm1d(d_mid))
# 			linear_hidden.append(nn.LeakyReLU(inplace=True))
# 		self.linear_hidden = nn.Sequential(*linear_hidden)
# 		self.linear_out = nn.Linear(d_inp if num_layers == 1 else d_mid, d_out) if num_layers >= 1 else nn.Identity()
# 		self.mlp_render = nn.Sequential(nn.Linear(d_out, self.dim_rend), nn.Tanh())

# 	def forward(self, x):
# 		x = self.linear_hidden(x)
# 		x = self.linear_out(x)
# 		return F.normalize(x, p=2, dim=-1)

# 	def render(self, feat, prob, margin=0.15):
# 		B, C, H, W = feat.shape
# 		with torch.no_grad():#渲染时候投影模块去除梯度，但渲染模块需要学习
# 			mask = ((prob.detach()-0.5).abs()<margin).type(torch.bool)
# 			feat = feat.permute(0,2,3,1).reshape(-1,C)[mask.permute(0,2,3,1).reshape(-1,1).repeat(1,C)]
# 			# print('feat:', feat.shape)
# 			proj = self.forward(feat.reshape(-1,C))
# 			# proj = self.tsne.fit_transform(rend)
# 			# print(prob.shape, mask.shape, proj.shape, proj.numel())
# 			rend = torch.zeros(B, self.dim_rend, H, W).to(feat.device)
# 			rend[mask.repeat(1,self.dim_rend,1,1)] = self.mlp_render(proj).view(-1)
# 		return rend

#start#
class ConvBlock(nn.Module):
	MyConv = BasicConv2d
	def __init__(self, in_c, out_c, stride=1, pool=True, **args):
		super(ConvBlock, self).__init__()
		if pool:
			stride = 2
		self.conv1 = self.MyConv(in_c, out_c, kernel_size=3, stride=stride)
		self.conv2 = nn.Sequential(
			nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
			# DisOut(),#prob=0.2
			nn.BatchNorm2d(out_c)
		)
		self.relu = nn.LeakyReLU()
		downsample = None
		if pool or in_c!=out_c:
			downsample = nn.Sequential(
				nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),# if pool else \
				# nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
				# DisOut(),#prob=0.2
				nn.BatchNorm2d(out_c),
			)
		self.downsample = downsample

	def forward(self, x):
		residual = x
		if self.downsample is not None:
			residual = self.downsample(x)
		# print('Basic:', x.shape)
		out = self.conv1(x)
		out = self.conv2(out)
		# print(out.shape, residual.shape)
		out = self.relu(out + residual)
		# print(out.min().item(), out.max().item())
		return out

class UpsampleBlock(torch.nn.Module):
	def __init__(self, in_c, out_c, up_mode='transp_conv'):
		super(UpsampleBlock, self).__init__()
		block = []
		if up_mode == 'transp_conv':
			block.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2))
		elif up_mode == 'up_conv':
			block.append(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False))
			block.append(nn.Conv2d(in_c, out_c, kernel_size=1))
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
		block.append(nn.LeakyReLU())
		block.append(nn.BatchNorm2d(out_c))

		self.block = nn.Sequential(*block)

	def forward(self, x):
		out = self.block(x)
		return out

class UpConvBlock(torch.nn.Module):
	def __init__(self, in_c, out_c, k_sz=3, up_mode='up_conv', conv_bridge=False):
		super(UpConvBlock, self).__init__()
		self.conv_bridge = conv_bridge

		self.up_layer = UpsampleBlock(in_c, out_c, up_mode=up_mode)
		self.conv_layer = ConvBlock(2 * out_c, out_c, k_sz=k_sz, pool=False)
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

class DoU(nn.Module):
	__name__ = 'dou'
	use_render = False
	def __init__(self, in_c=1, n_classes=1, layers=(32,32,32,32,32)):
		super(DoU, self).__init__()
		self.num_features = layers[-1]

		self.__name__ = 'dou{}x{}'.format(len(layers), layers[0])
		self.n_classes = n_classes
		self.first = ConvBlock(in_c=in_c, out_c=layers[0], pool=False)

		self.down_path = nn.ModuleList()
		for i in range(len(layers) - 1):
			block = ConvBlock(in_c=layers[i], out_c=layers[i + 1], pool=True)
			self.down_path.append(block)

		self.up_path = nn.ModuleList()
		reversed_layers = list(reversed(layers))
		for i in range(len(layers) - 1):
			block = UpConvBlock(in_c=reversed_layers[i], out_c=reversed_layers[i + 1])
			self.up_path.append(block)

		self.projection = MlpNorm(32)
		self.out = OutSigmoid(1+self.projection.dim_rend, n_classes)
		self.aux = OutSigmoid(layers[0], n_classes)
		#nn.Sequential(
		# 	nn.Conv2d(layers[0], n_classes, kernel_size=1),
		# 	# nn.Conv2d(n_classes, n_classes, kernel_size=1),
		# 	nn.BatchNorm2d(n_classes),
		# 	nn.Sigmoid()
		# )

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
		# self.feat = x
		# print(x.shape)

		aux = self.aux(x)
		self.pred = aux
		if self.use_render:
			rend = self.projection.render(self.feat, aux)
			out = self.out(torch.cat([aux, rend], dim=1))
			self.pred = out
			return out, aux
		return self.pred
#end#


if __name__ == '__main__':
	net = DoU()
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