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

def torch_dilation(x, ksize=5, stride=1):
	return F.max_pool2d(x, (ksize, ksize), stride, ksize//2)

class MorphBlock(nn.Module):
    def __init__(self, channel=8):
        super().__init__()
        self.ch_wv = nn.Sequential(
            nn.Conv2d(2,channel,kernel_size=5, padding=2),
            nn.Conv2d(channel,channel,kernel_size=5, padding=2),
            nn.Conv2d(channel,channel//2,kernel_size=3, padding=1),
        )
        self.ch_wq = nn.Sequential(
            nn.Conv2d(channel//2,8,kernel_size=3, padding=1),
            nn.Conv2d(8,1,kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = torch.cat([torch_dilation(x, ksize=3), x], dim=1)#, 1-torch_dilation(1-x, ksize=3)
        return self.ch_wq(self.ch_wv(x))
		
class MlpNorm(nn.Module):
	def __init__(self, dim_inp=256, dim_out=64):
		super(MlpNorm, self).__init__()
		dim_mid = min(dim_inp, dim_out)#max(dim_inp, dim_out)//2
		# hidden layers
		linear_hidden = []#nn.Identity()
		# for i in range(num_layers - 1):
		linear_hidden.append(nn.Linear(dim_inp, dim_mid))
		# linear_hidden.append(nn.BatchNorm1d(dim_mid))
		linear_hidden.append(nn.Dropout(p=0.2))
		linear_hidden.append(nn.LeakyReLU())
		self.linear_hidden = nn.Sequential(*linear_hidden)

		self.linear_out = nn.Linear(dim_mid, dim_out)# if num_layers >= 1 else nn.Identity()

	def forward(self, x):
		x = self.linear_hidden(x)
		x = self.linear_out(x)
		return F.normalize(x, p=2, dim=-1)

class MlpSphere(nn.Module):#球极平面射影
	def __init__(self, dim_inp=64, dim_out=32):
		super(MlpSphere, self).__init__()
		# dim_mid = max(dim_inp, dim_out)//2
		# hidden layers
		linear_hidden = []#nn.Identity()
		# for i in range(num_layers - 1):
		linear_hidden.append(nn.Linear(dim_inp, dim_out))
		# linear_hidden.append(nn.BatchNorm1d(dim_out))
		linear_hidden.append(nn.Dropout(p=0.2))
		linear_hidden.append(nn.LeakyReLU())
		self.linear_hidden = nn.Sequential(*linear_hidden)

		# self.linear_out = OutSigmoid(dim_out, 1)

	def forward(self, x):
		x = self.linear_hidden(x)
		# B,C,H,W = x.shape
		# x = self.linear_hidden(x.view(B,C,-1).permute(0,2,1))
		# x = self.linear_out(x.permute(0,2,1).view(B,-1,H,W))
		return x

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

	def regular(self, sampler, lab, fov=None, sim=True):
		feat = sampler.select(self.feat, self.pred.detach(), lab, fov)
		if sim:
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
	
def sbau(**args):
	ConvBlock.attention='spolar'
	ConvBlock.MyConv = BasicConv2d
	net = LUNet(**args)
	net.__name__ = 'sbau'
	return net
def sdou(**args):
	ConvBlock.attention='spolar'
	ConvBlock.MyConv = DoverConv2d
	net = LUNet(**args)
	net.__name__ = 'sdou'
	return net
def spyu(**args):
	ConvBlock.attention='spolar'
	ConvBlock.MyConv = PyridConv2d
	net = LUNet(**args)
	net.__name__ = 'spyu'
	return net
def sdxu(**args):
	ConvBlock.attention='spolar'
	ConvBlock.MyConv = DiffXConv2d
	net = LUNet(**args)
	net.__name__ = 'sdxu'
	return net
#end#



if __name__ == '__main__':
	import time

	net = lunet()
	net = munet()
	# net = punet()
	# net = sunet()
	# net = sdxu()
	# net = spyu()
	# net = sbau()
	# net = sdou()
	net.use_render = True


	x = torch.rand(2,1,64,64)

	st = time.time()
	ys = net(x)
	print('time:', time.time()-st)

	for y in ys:
		print('pred:', y.shape)
	# print(net.__name__, y['loss'])

	sampler = MLPSampler(top=4, low=0, mode='hard')
	# net.train()
	l = net.regular(sampler, torch.rand(2,1,64,64), torch.rand(2,1,64,64), sim=False)
	print(net.__name__, l.item())
	l = net.regular(sampler, torch.rand(2,1,64,64), torch.rand(2,1,64,64), sim=True)
	print(net.__name__, l.item())

	# plot(net.emb)
	print('Params model:',sum(p.numel() for p in net.parameters() if p.requires_grad))