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

import sys
sys.path.append('.')
sys.path.append('..')

from nets import *
from utils import *

#start#
import monai
def swish(x):
	# return x * torch.sigmoid(x)   #计算复杂
	return x * F.relu6(x+3)/6       #计算简单

class LUConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, 
				stride=1, padding=1, dilation=1, groups=1, bias=False, bn=True, 
				out='dis', activation=swish, conv=nn.Conv2d, 
				):#'frelu',nn.ReLU(inplace=False),sinlu
		super(LUConv2d, self).__init__()
		if not isinstance(kernel_size, tuple):
			if dilation>1:
				padding = dilation*(kernel_size//2)	#AtrousConv2d
			elif kernel_size==stride:
				padding=0
			else:
				padding = kernel_size//2			#LUConv2d

		self.c = conv(in_channels, out_channels, 
			kernel_size=kernel_size, stride=stride, 
			padding=padding, dilation=dilation, bias=bias)
		self.b = nn.BatchNorm2d(out_channels, momentum=0.01) if bn else nn.Identity()

		self.o = nn.Identity()
		drop_prob=0.15
		# self.o = DisOut(drop_prob=drop_prob)#
		self.o = nn.Dropout2d(p=drop_prob, inplace=False) 

		if activation=='frelu':
			self.a = FReLU(out_channels)
		elif activation is None:
			self.a = nn.Identity()
		else:
			self.a = activation

	def forward(self, x):
		x = self.c(x)# x = torch.clamp_max(x, max=99)
		# print('c:', x.max().item())
		x = self.o(x)
		# print('o:', x.max().item())
		x = self.b(x)
		# print('b:', x.max().item())
		x = self.a(x)
		# print('a:', x.max().item())
		return x

class LUBlock(torch.nn.Module):
	attention=None
	MyConv = LUConv2d
	def __init__(self, inp_c, out_c, ksize=3, shortcut=False, pool=True):
		super(LUBlock, self).__init__()
		self.shortcut = nn.Sequential(nn.Conv2d(inp_c, out_c, kernel_size=1), nn.BatchNorm2d(out_c))
		pad = (ksize - 1) // 2

		if pool: self.pool = nn.MaxPool2d(kernel_size=2)
		else: self.pool = False

		block = []
		block.append(self.MyConv(inp_c, out_c, kernel_size=ksize, padding=pad))
		block.append(self.MyConv(out_c, out_c, kernel_size=ksize, padding=pad))
		self.block = nn.Sequential(*block)
	def forward(self, x):
		if self.pool: x = self.pool(x)
		out = self.block(x)
		return swish(out + self.shortcut(x))

# 输出层 & 下采样
class OutSigmoid(nn.Module):
	def __init__(self, inp_planes, out_planes=1, out_c=8):
		super(OutSigmoid, self).__init__()
		self.cls = nn.Sequential(
			nn.Conv2d(in_channels=inp_planes, out_channels=out_c, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_c),
			nn.Conv2d(in_channels=out_c, out_channels=1, kernel_size=1, bias=True),
			nn.Sigmoid()
		)
	def forward(self, x):
		return self.cls(x)

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

class UpLUBlock(torch.nn.Module):
	def __init__(self, inp_c, out_c, ksize=3, up_mode='up_conv', conv_bridge=False, shortcut=False):
		super(UpLUBlock, self).__init__()
		self.conv_bridge = conv_bridge

		self.up_layer = UpsampleBlock(inp_c, out_c, up_mode=up_mode)
		self.conv_layer = LUBlock(2 * out_c, out_c, ksize=ksize, shortcut=shortcut, pool=False)
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

class LUBasicConv2d(nn.Module):
	def __init__(self, inp_c, out_c, kernel_size=3, 
				stride=1, padding=1, dilation=1, groups=1, bias=False, bn=True, 
				activation=swish, conv=nn.Conv2d, dropp=0.15
				):#'frelu',nn.ReLU(inplace=False),sinlu
		super(LUBasicConv2d, self).__init__()
		if not isinstance(kernel_size, tuple):
			if dilation>1:
				padding = dilation*(kernel_size//2)	#AtrousConv2d
			elif kernel_size==stride:
				padding=0
			else:
				padding = kernel_size//2			#LUBasicConv2d

		self.c = conv(inp_c, out_c, 
				kernel_size=kernel_size, stride=stride, 
				padding=padding, dilation=dilation, bias=bias)

		if activation is None:
			self.a = nn.Sequential()
		else:
			self.a = activation
		
		self.b = nn.BatchNorm2d(out_c) if bn else nn.Sequential()
		# self.b = nn.InstanceNorm2d(out_c) if bn else nn.Sequential()
		self.o = nn.Dropout2d(p=dropp)#DisOut(p=.15)#
	
	def forward(self, x):
		x = self.c(x)
		x = self.b(x)
		x = self.o(x)
		x = self.a(x)
		return x

class PoolNeck(torch.nn.Module):
	MyConv = LUBasicConv2d
	def __init__(self, phase, ksize=3, out='dis', **args):
		super(PoolNeck, self).__init__()
		# padding = ksize-1#(ksize-1)//2
		# print(ksize, padding)
		self.shortcut = nn.Sequential(
			nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
			nn.Conv2d(phase, phase, kernel_size=3, stride=1, padding=1, bias=False, groups=phase), 
			nn.BatchNorm2d(phase)

			# nn.MaxPool2d(kernel_size=5, stride=3, padding=2),
			# nn.Conv2d(phase, phase, kernel_size=3, stride=1, padding=1, bias=False, groups=phase), 
			
			# nn.Conv2d(phase, phase, kernel_size=5, stride=3, padding=2, bias=False), 
			# nn.Conv2d(phase, 1, kernel_size=3, stride=1, padding=1, bias=False), 
			# nn.BatchNorm2d(1)
			)
			
		self.conv1 = self.MyConv(phase, phase, ksize, padding=1, groups=phase)
		self.conv2 = self.MyConv(phase, phase, ksize, padding=1, activation=None, groups=phase)
		self.conv3 = nn.Conv2d(phase, phase, 1, 1, 0, groups=phase)
	def forward(self, x):
		out = self.conv2(self.conv1(x))
		# out = self.o(out)
		residual = self.shortcut(x)
		# residual = F.interpolate(residual, size=out.shape[-2:], mode='bilinear', align_corners=False)
		# print(out.shape, residual.shape)
		# return torch.relu(out + residual)
		return self.conv3(out + residual)

class SMFU(nn.Module):
	__name__ = 'smfu'
	# def __init__(self, inp_c=1, n_classes=1, layers=(16,20,24,28,32)):
	def __init__(self, inp_c=1, width=32, depth=5, scale=5, *args, **kwargs):
		super(SMFU, self).__init__()
		layers = [width,]*5
		self.num_features = layers[-1]

		num_con=width
		# self.__name__ += 'w{}d{}'.format(width, depth)
		self.scale = scale
		self.channels = width
		# self.rot3 = Rotation(channels=width, ksize=3)
		self.rot5 = Rotation(channels=width, ksize=5)
		# self.rot7 = Rotation(channels=width, ksize=7)
		# self.rot9 = Rotation(channels=width, ksize=9)
		for p in self.rot5.parameters():
			p.requires_grad=False
		self.beta0 = nn.Parameter(torch.ones(size=(1,), dtype=torch.float32))
		self.beta1 = nn.Parameter(torch.ones(size=(1,), dtype=torch.float32))
		self.beta2 = nn.Parameter(torch.ones(size=(1,), dtype=torch.float32))
		self.beta3 = nn.Parameter(torch.ones(size=(1,), dtype=torch.float32))
		self.beta4 = nn.Parameter(torch.ones(size=(1,), dtype=torch.float32))
		self.pooling = nn.MaxPool2d(3,2,1)

		# self.buff_std = LUBasicConv2d(inp_c=scale, out_c=num_con)
		# self.buff_vpp = LUBasicConv2d(inp_c=scale, out_c=num_con)
		self.segfirst = LUBasicConv2d(inp_c=inp_c, out_c=width)
		# self.segfinal = OutSigmoid(num_con)

		self.conv_cat = nn.Sequential(
			nn.Conv2d(2,1,1,1,0),
			nn.BatchNorm2d(1)
		)
		# self.conv_cat = OutSigmoid(2,1)
		self.projector = None
		self.predictor = None

		self.encoders = nn.ModuleList()
		self.divisits = nn.ModuleList()

		#	用序列Block的深度模仿：匹配滤波核的尺寸变化
		for _ in range(scale):
			encoder = []
			for _ in range(depth):
				block = PoolNeck(phase=width, ksize=5)
				# block = BottleNeck(phase=width)
				# block = NeXtNeck(phase=width)
				encoder.append(block)
			encoder = nn.Sequential(*encoder)
			self.divisits.append(OutSigmoid(width))
			self.encoders.append(encoder)

		self.__name__ = 'u{}x{}'.format(len(layers), layers[0])

		self.up_path = nn.ModuleList()
		for i in range(len(layers)):
			block = UpLUBlock(inp_c=width, out_c=width)
			self.up_path.append(block)

		self.out = OutSigmoid(layers[0], 1)

	def regular_rot(self):
		losSum = []
		for w in self.encoders.parameters():
			# print('regular_rot:', w.shape)
		# for m in self.modules():
			# print(w.shape)
		# 	if isinstance(m, nn.Conv2d):
			if len(w.shape)==4 and w.shape[-1]>=3 and w.shape[1]==self.channels and w.shape[0]==self.channels: 
				# print('regular_rot:', w.shape)
				los = eval('self.rot'+str(w.shape[-1]))(w)
				losSum.append(los)
		losSum = sum(losSum) / len(losSum)
		return losSum
	
	def regular_bce(self, **args):
		return sum(self.bces_list)

	def soft_argmax(self, x, beta=100):
		# x = x.reshape(x.shape[0], x.shape[1], -1)#[b,c,h,w]
		soft_max = F.softmax(x*beta, dim=1).view(x.shape).clamp(0,1)
		f_space = torch.std(soft_max, dim=1, keepdim=True)
		hard_tgt = monai.networks.utils.one_hot(soft_max.argmax(dim=1, keepdim=True), num_classes=soft_max.shape[1], dim=1)
		loss_entropy = F.binary_cross_entropy(soft_max, hard_tgt, weight=f_space.detach()) #* f_space.detach()#理应只在血管处有各向异性（朝向感知）
		# print('regular_bce:', loss_entropy.shape, f_space.shape)
		self.bces_list.append(loss_entropy)
		# b,c,h,w = x.shape
		# std = self.orientor(soft_max.permute(0,2,3,1).reshape(-1,c)).reshape(b,h,w,1).permute(0,3,1,2)
		# return std

		# soft_max = F.gumbel_softmax(x, dim=1, tau=beta)
		idx_weight = torch.arange(start=0, end=x.shape[1]).reshape(1,-1,1,1)
		# print(x.shape, soft_max.shape, idx_weight.shape)
		matmul = soft_max * idx_weight.to(soft_max.device)
		f_orientation = matmul.sum(dim=1, keepdim=True)
		return self.conv_cat(torch.cat([f_space, f_orientation], dim=1)) + x

	tmp={}
	flag_down = False
	bces_list=[]
	def forward(self, x):
		h,w = x.shape[-2:]
		if self.flag_down:
			x = F.interpolate(x, size=(h//2,w//2), mode='bilinear', align_corners=False)
		# x,e = torch.chunk(x, 2, dim=1)
		x = self.segfirst(x)
		down_activations = []
		# print('x:', x.shape)
		# stds = []
		# beta = 10**(torch.tanh(self.beta)+2)
		# print('beta:', beta)
		self.bces_list=[]
		for i in range(self.scale):
			self.tmp['smf'+str(i)] = x
			down_activations.append(x.clone())
			# print('x:', x.shape)
			x = self.encoders[i](x)
			x = self.soft_argmax(x, beta=eval('self.beta'+str(i)))
			# stds.append(x.clone())
			x = self.pooling(x)
		down_activations.reverse()

		auxs = []
		for i, up in enumerate(self.up_path):
			x = up(x, down_activations[i])
			sig = self.divisits[i](x)#
			auxs.append(sig)

		self.feat = x
		y = self.out(x)
		# auxs.append(y)
		# y = self.unet(x)
		self.pred = y
		return self.pred

		# auxs.append(y)
		# auxs.reverse()
		# if self.flag_down:
		# 	self.pred = F.interpolate(self.pred, size=(h,w), mode='bilinear', align_corners=False)
		# for i in range(len(auxs)):
		# 	auxs[i] = F.interpolate(auxs[i], size=(h,w), mode='bilinear', align_corners=False)
		# return auxs

def smf(**args):
	net = SMFU(**args)
	net.__name__ = 'smf'
	return net
#end#



if __name__ == '__main__':
	import time

	net = smf(inp_c=1)


	x = torch.rand(1,1,64,64)

	st = time.time()
	ys = net(x)
	print('time:', time.time()-st)
	for y in ys:
		print('pred:', y.shape, y.min().item(), y.max().item())
	print(net.__name__, net.feat.shape, net.regular_bce())

	# plot(net.emb)
	print('Params model:',sum(p.numel() for p in net.parameters() if p.requires_grad))
