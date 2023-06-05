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

#start#
def swish(x):
	# return x * torch.sigmoid(x)   #计算复杂
	return x * F.relu6(x+3)/6       #计算简单

class FReLU(nn.Module):
	r""" FReLU formulation, with a window size of kxk. (k=3 by default)"""
	def __init__(self, in_channels, *args):
		super().__init__()
		self.func = nn.Sequential(
			nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
			nn.BatchNorm2d(in_channels)
		)
	def forward(self, x):
		x = torch.max(x, self.func(x))
		return x

class DisOut(nn.Module):
	def __init__(self, drop_prob=0.5, block_size=6, alpha=1.0):
		super(DisOut, self).__init__()

		self.drop_prob = drop_prob      
		self.weight_behind=None
  
		self.alpha=alpha
		self.block_size = block_size
		
	def forward(self, x):
		if not self.training: 
			return x

		x=x.clone()
		if x.dim()==4:           
			width=x.size(2)
			height=x.size(3)

			seed_drop_rate = self.drop_prob* (width*height) / self.block_size**2 / (( width -self.block_size + 1)*( height -self.block_size + 1))
			
			valid_block_center=torch.zeros(width,height,device=x.device).float()
			valid_block_center[int(self.block_size // 2):(width - (self.block_size - 1) // 2),int(self.block_size // 2):(height - (self.block_size - 1) // 2)]=1.0
			valid_block_center=valid_block_center.unsqueeze(0).unsqueeze(0)
			
			randdist = torch.rand(x.shape,device=x.device)
			block_pattern = ((1 -valid_block_center + float(1 - seed_drop_rate) + randdist) >= 1).float()
		
			if self.block_size == width and self.block_size == height:            
				block_pattern = torch.min(block_pattern.view(x.size(0),x.size(1),x.size(2)*x.size(3)),dim=2)[0].unsqueeze(-1).unsqueeze(-1)
			else:
				block_pattern = -F.max_pool2d(input=-block_pattern, kernel_size=(self.block_size, self.block_size), stride=(1, 1), padding=self.block_size // 2)

			if self.block_size % 2 == 0:
					block_pattern = block_pattern[:, :, :-1, :-1]
			percent_ones = block_pattern.sum() / float(block_pattern.numel())

			if not (self.weight_behind is None) and not(len(self.weight_behind)==0):
				wtsize=self.weight_behind.size(3)
				weight_max=self.weight_behind.max(dim=0,keepdim=True)[0]
				sig=torch.ones(weight_max.size(),device=weight_max.device)
				sig[torch.rand(weight_max.size(),device=sig.device)<0.5]=-1
				weight_max=weight_max*sig 
				weight_mean=weight_max.mean(dim=(2,3),keepdim=True)
				if wtsize==1:
					weight_mean=0.1*weight_mean
				#print(weight_mean)
			mean=torch.mean(x).clone().detach()
			var=torch.var(x).clone().detach()

			if not (self.weight_behind is None) and not(len(self.weight_behind)==0):
				dist=self.alpha*weight_mean*(var**0.5)*torch.randn(*x.shape,device=x.device)
			else:
				dist=self.alpha*0.01*(var**0.5)*torch.randn(*x.shape,device=x.device)

		x=x*block_pattern
		dist=dist*(1-block_pattern)
		x=x+dist
		x=x/percent_ones
		return x

import math
import torch
import numpy as np
from torch.nn import init
from itertools import repeat
from torch.nn import functional as F
from torch._six import container_abcs
from torch._jit_internal import Optional
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class DOConv2d(Module):
	"""
	   DOConv2d can be used as an alternative for torch.nn.Conv2d.
	   The interface is similar to that of Conv2d, with one exception:
			1. D_mul: the depth multiplier for the over-parameterization.
	   Note that the groups parameter switchs between DO-Conv (groups=1),
	   DO-DConv (groups=in_channels), DO-GConv (otherwise).
	"""
	__constants__ = ['stride', 'padding', 'dilation', 'groups',
					 'padding_mode', 'output_padding', 'in_channels',
					 'out_channels', 'kernel_size', 'D_mul']
	__annotations__ = {'bias': Optional[torch.Tensor]}

	def __init__(self, in_channels, out_channels, kernel_size, D_mul=None, stride=1,
				 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
		super(DOConv2d, self).__init__()

		kernel_size = _pair(kernel_size)
		stride = _pair(stride)
		padding = _pair(padding)
		dilation = _pair(dilation)

		if in_channels % groups != 0:
			raise ValueError('in_channels must be divisible by groups')
		if out_channels % groups != 0:
			raise ValueError('out_channels must be divisible by groups')
		valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
		if padding_mode not in valid_padding_modes:
			raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
				valid_padding_modes, padding_mode))
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.dilation = dilation
		self.groups = groups
		self.padding_mode = padding_mode
		self._padding_repeated_twice = tuple(x for x in self.padding for _ in range(2))

		#################################### Initailization of D & W ###################################
		M = self.kernel_size[0]
		N = self.kernel_size[1]
		self.D_mul = M * N if D_mul is None or M * N <= 1 else D_mul
		self.W = Parameter(torch.Tensor(out_channels, in_channels // groups, self.D_mul))
		init.kaiming_uniform_(self.W, a=math.sqrt(5))

		if M * N > 1:
			self.D = Parameter(torch.Tensor(in_channels, M * N, self.D_mul))
			init_zero = np.zeros([in_channels, M * N, self.D_mul], dtype=np.float32)
			self.D.data = torch.from_numpy(init_zero)

			eye = torch.reshape(torch.eye(M * N, dtype=torch.float32), (1, M * N, M * N))
			D_diag = eye.repeat((in_channels, 1, self.D_mul // (M * N)))
			if self.D_mul % (M * N) != 0:  # the cases when D_mul > M * N
				zeros = torch.zeros([in_channels, M * N, self.D_mul % (M * N)])
				self.D_diag = Parameter(torch.cat([D_diag, zeros], dim=2), requires_grad=False)
			else:  # the case when D_mul = M * N
				self.D_diag = Parameter(D_diag, requires_grad=False)
		##################################################################################################

		if bias:
			self.bias = Parameter(torch.Tensor(out_channels))
			fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
			bound = 1 / math.sqrt(fan_in)
			init.uniform_(self.bias, -bound, bound)
		else:
			self.register_parameter('bias', None)

	def extra_repr(self):
		s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
			 ', stride={stride}')
		if self.padding != (0,) * len(self.padding):
			s += ', padding={padding}'
		if self.dilation != (1,) * len(self.dilation):
			s += ', dilation={dilation}'
		if self.groups != 1:
			s += ', groups={groups}'
		if self.bias is None:
			s += ', bias=False'
		if self.padding_mode != 'zeros':
			s += ', padding_mode={padding_mode}'
		return s.format(**self.__dict__)

	def __setstate__(self, state):
		super(DOConv2d, self).__setstate__(state)
		if not hasattr(self, 'padding_mode'):
			self.padding_mode = 'zeros'

	def _conv_forward(self, input, weight):
		if self.padding_mode != 'zeros':
			return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
							weight, self.bias, self.stride,
							_pair(0), self.dilation, self.groups)
		return F.conv2d(input, weight, self.bias, self.stride,
						self.padding, self.dilation, self.groups)

	def forward(self, input):
		M = self.kernel_size[0]
		N = self.kernel_size[1]
		DoW_shape = (self.out_channels, self.in_channels // self.groups, M, N)
		if M * N > 1:
			######################### Compute DoW #################
			# (input_channels, D_mul, M * N)
			D = self.D + self.D_diag
			W = torch.reshape(self.W, (self.out_channels // self.groups, self.in_channels, self.D_mul))

			# einsum outputs (out_channels // groups, in_channels, M * N),
			# which is reshaped to
			# (out_channels, in_channels // groups, M, N)
			DoW = torch.reshape(torch.einsum('ims,ois->oim', D, W), DoW_shape)
			#######################################################
		else:
			# in this case D_mul == M * N
			# reshape from
			# (out_channels, in_channels // groups, D_mul)
			# to
			# (out_channels, in_channels // groups, M, N)
			DoW = torch.reshape(self.W, DoW_shape)
		return self._conv_forward(input, DoW)

def _ntuple(n):
	def parse(x):
		if isinstance(x, container_abcs.Iterable):
			return x
		return tuple(repeat(x, n))

	return parse

_pair = _ntuple(2)

def conv(in_out_channels, out_out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
	"""standard convolution with padding"""
	return nn.Conv2d(in_out_channels, out_out_channels, kernel_size=kernel_size, stride=stride,
					 padding=padding, dilation=dilation, groups=groups, bias=False)

class PyConv3(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1, pyconv_kernels=[3, 5, 7], pyconv_groups=[1, 4, 8], **args):
		super(PyConv3, self).__init__()
		self.conv2_1 = conv(in_channels, out_channels // 4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
							stride=stride, groups=pyconv_groups[0])
		self.conv2_2 = conv(in_channels, out_channels // 4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
							stride=stride, groups=pyconv_groups[1])
		self.conv2_3 = conv(in_channels, out_channels // 2, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2] // 2,
							stride=stride, groups=pyconv_groups[2])

	def forward(self, x):
		return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x)), dim=1)

class GarborConv2(nn.Module):#Coarse to Fine Convlution
	#   感受野可以试试sigmoid，试试BatchNorm Addition
	def __init__(self, in_channels, out_channels, stride=1, **args):
		super(GarborConv2, self).__init__()
		self.gauss = nn.Parameter(torch.randn(size=(out_channels, in_channels, 3, 3)), requires_grad=True)
		self.complex = nn.Parameter(torch.randn(size=(out_channels, in_channels, 3, 3)), requires_grad=True)
		self.stride = stride
	def forward(self, x):
		gauss = torch.clamp_max(self.gauss, 0.4)
		weight = gauss * torch.cos(self.complex)
		return  F.conv2d(x, weight, stride=self.stride, padding=self.stride%2)

class SCNetConv2d(nn.Module):
	# 作者团队：南开大学(程明明组)b&NUS&字节AI Lab
	# http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf
	# 代码链接：https://github.com/backseason/SCNet
	# SCNet：通过自校准卷积改进卷积网络
	def __init__(self, in_channels, out_channels, pooling_ratio=2, stride=1, **kwargs):
		super(SCNetConv2d, self).__init__()
		self.k2 = nn.Sequential(
			nn.AvgPool2d(kernel_size=pooling_ratio, stride=pooling_ratio),
			nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
			nn.BatchNorm2d(in_channels)
		)
		self.k3 = nn.Sequential(
			nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
			nn.BatchNorm2d(in_channels)
		)
		self.k4 = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(),
		)
	def forward(self, x):
		out = torch.sigmoid(x + F.interpolate(self.k2(x), x.size()[2:]))
		out = torch.mul(self.k3(x), out)
		return self.k4(out)

class SpAttConv2d(nn.Module):
	"""Split-Attention nn.Conv2d . Idea from ResNeSt"""
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
				dilation=1, groups=1, radix=2, reduction_factor=4, **kwargs):
		super(SpAttConv2d, self).__init__()
		inter_channels = max(in_channels*radix//reduction_factor, 32)
		self.radix = radix
		self.cardinality = groups
		self.out_channels = out_channels
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels*radix, kernel_size, stride, padding, dilation,
							groups=groups*radix, bias=False),
			nn.BatchNorm2d(out_channels*radix),
			DisOut(),
			nn.ReLU(),
		)
		self.fc1 = nn.Sequential(
			nn.Conv2d(out_channels, inter_channels, 1, groups=self.cardinality),
			nn.BatchNorm2d(inter_channels),#由于BatchNorm操作需要多于一个数据计算平均值
			nn.ReLU(),
		)
		self.fc2 = nn.Conv2d(inter_channels, out_channels*radix, 1, groups=self.cardinality)

	def forward(self, x):
		x = self.conv(x)

		batch, channel = x.shape[:2]
		if self.radix > 1:
			splited = torch.split(x, channel//self.radix, dim=1)
			gap = sum(splited) 
		else:
			gap = x
		gap = F.adaptive_avg_pool2d(gap, 1)
		gap = self.fc1(gap)

		atten = self.fc2(gap).view((batch, self.radix, self.out_channels))
		if self.radix > 1:
			atten = F.softmax(atten, dim=1).view(batch, -1, 1, 1)
		else:
			atten = torch.sigmoid(atten).view(batch, -1, 1, 1)

		if self.radix > 1:
			atten = torch.split(atten, channel//self.radix, dim=1)
			out = sum([att*split for (att, split) in zip(atten, splited)])
		else:
			out = atten * x
		return out.contiguous()

class BasicConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, 
				stride=1, padding=1, dilation=1, groups=1, bias=False, bn=True, 
				out='dis', activation=swish, conv=nn.Conv2d, 
				):#'frelu',nn.ReLU(inplace=False),sinlu
		super(BasicConv2d, self).__init__()
		if not isinstance(kernel_size, tuple):
			if dilation>1:
				padding = dilation*(kernel_size//2)	#AtrousConv2d
			elif kernel_size==stride:
				padding=0
			else:
				padding = kernel_size//2			#BasicConv2d

		self.c = conv(in_channels, out_channels, 
			kernel_size=kernel_size, stride=stride, 
			padding=padding, dilation=dilation, bias=bias)
		self.b = nn.BatchNorm2d(out_channels, momentum=0.01) if bn else nn.Sequential()

		self.o = nn.Sequential()
		drop_prob=0.15
		if out=='dis':
			self.o = DisOut(drop_prob=drop_prob)#
		elif out=='spatial':
			self.o = SpatialDropout(drop_prob=drop_prob) 
		elif out=='target':
			self.o = TargetDrop(channels=out_channels, drop_prob=drop_prob)
		elif out=='drop':
			self.o = nn.Dropout2d(p=drop_prob, inplace=False) 

		if activation=='frelu':
			self.a = FReLU(out_channels)
		elif activation is None:
			self.a = nn.Sequential()
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

class PyridConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, 
				stride=1, padding=1, dilation=1, groups=1, bias=False, bn=True, 
				):#ACTIVATION,nn.PReLU(),torch.sin
		super(PyridConv2d, self).__init__()
		self.c = BasicConv2d(in_channels=in_channels, out_channels=out_channels, 
				stride=stride, bn=True, conv=PyConv3, groups=in_channels)
	def forward(self, x):
		return self.c(x)

class DoverConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, 
				stride=1, padding=1, dilation=1, groups=1, bias=False, bn=True, 
				out='dis', activation=F.gelu,
				):#ACTIVATION,nn.PReLU(),torch.sin
		super(DoverConv2d, self).__init__()
		self.c = BasicConv2d(in_channels=in_channels, out_channels=out_channels, 
				stride=stride, bn=True, conv=DOConv2d)
	def forward(self, x):
		return self.c(x)

class Bottleneck(nn.Module):
	MyConv = BasicConv2d
	def __init__(self, in_c, out_c, stride=1, downsample=None, **args):
		super(Bottleneck, self).__init__()

		self.conv1 = self.MyConv(in_c, out_c, kernel_size=3, stride=stride)
		self.conv2 = nn.Sequential(
			nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
			# DisOut(),#prob=0.2
			nn.BatchNorm2d(out_c)
		)
		self.relu = nn.LeakyReLU()
		if downsample is None and in_c != out_c:
			downsample = nn.Sequential(
				nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride),
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
		out = self.relu(out + residual)
		# print(out.min().item(), out.max().item())
		return out

# 输出层 & 下采样
class OutSigmoid(nn.Module):
	def __init__(self, inp_planes, out_planes=1, out_c=8):
		super(OutSigmoid, self).__init__()
		self.cls = nn.Sequential(
			nn.Conv2d(in_channels=inp_planes, out_channels=out_c, kernel_size=3, padding=1, bias=False),
			# nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_c),
			nn.Conv2d(in_channels=out_c, out_channels=1, kernel_size=1, bias=True),
			nn.Sigmoid()
		)
	def forward(self, x):
		return self.cls(x)

class HRModule(nn.Module):
	def __init__(self, num_branches, num_blocks, num_inchannels, num_channels, multi_scale_output=True):
		super(HRModule, self).__init__()
		self.num_inchannels = num_inchannels
		self.num_branches = num_branches
		self.multi_scale_output = multi_scale_output

		self.branches = self._make_branches(num_branches, num_blocks, num_channels)
		self.fuse_layers = self._make_fuse_layers()
		self.relu = nn.PReLU()

	def _make_one_branch(self, branch_index, num_blocks, num_channels, stride=1):
		downsample = None
		if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index]:
			downsample = nn.Sequential(
				nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index],
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(num_channels[branch_index]),
			)

		layers = []
		layers.append(Bottleneck(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample))
		self.num_inchannels[branch_index] = num_channels[branch_index]
		for i in range(1, num_blocks[branch_index]):
			layers.append(Bottleneck(self.num_inchannels[branch_index], num_channels[branch_index]))
		return nn.Sequential(*layers)

	def _make_branches(self, num_branches, num_blocks, num_channels):
		branches = []
		for i in range(num_branches):
			branches.append(self._make_one_branch(i, num_blocks, num_channels))
		return nn.ModuleList(branches)

	def _make_fuse_layers(self):
		if self.num_branches == 1:
			return None

		num_branches = self.num_branches
		num_inchannels = self.num_inchannels
		fuse_layers = []
		for i in range(num_branches if self.multi_scale_output else 1):
			fuse_layer = []
			for j in range(num_branches):
				if j > i:
					fuse_layer.append(nn.Sequential(
						nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
						nn.BatchNorm2d(num_inchannels[i])))
				elif j == i:
					fuse_layer.append(None)
				else:
					conv3x3s = []
					for k in range(i-j):
						if k == i - j - 1:
							num_outchannels_conv3x3 = num_inchannels[i]
							conv3x3s.append(nn.Sequential(
								nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
								nn.BatchNorm2d(num_outchannels_conv3x3)))
						else:
							num_outchannels_conv3x3 = num_inchannels[j]
							conv3x3s.append(nn.Sequential(
								nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
								nn.BatchNorm2d(num_outchannels_conv3x3),
								nn.PReLU()))
					fuse_layer.append(nn.Sequential(*conv3x3s))
			fuse_layers.append(nn.ModuleList(fuse_layer))
		return nn.ModuleList(fuse_layers)

	def forward(self, x):
		if self.num_branches == 1:
			return [self.branches[0](x[0])]

		for i in range(self.num_branches):
			x[i] = self.branches[i](x[i])

		x_fuse = []
		for i in range(len(self.fuse_layers)):
			y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
			for j in range(1, self.num_branches):
				if i == j:
					y = y + x[j]
				elif j > i:
					y = y + F.interpolate(self.fuse_layers[i][j](x[j]),
						size=x[i].shape[-2:], mode='bilinear', align_corners=True)
				else:
					y = y + self.fuse_layers[i][j](x[j])
			x_fuse.append(self.relu(y))
		return x_fuse

class HRNet(nn.Module):
	__name__ = 'hrnet'
	MyConv = BasicConv2d
	def __init__(self, extra, inp_planes=2, phase=32, name='hrba',  **kwargs):
		super(HRNet, self).__init__()
		self.__name__ = name
		self.num_features = extra['STAGE2']['NUM_CHANNELS'] [0]
		Bottleneck.MyConv = self.MyConv
		print('Conv Style of HRNet:', str(self.MyConv.__name__))

		# stem net
		self.layer0 = nn.Sequential(
			nn.Conv2d(inp_planes, phase, kernel_size=3, stride=1, padding=1, bias=False),
			nn.Conv2d(phase, phase, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(phase),
			# nn.PReLU()
		)
		
		self.stage1_cfg = extra['STAGE1']
		num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
		num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
		self.layer1 = self._make_layer(phase, num_channels, num_blocks)
		stage1_out_channel = num_channels

		self.stage2_cfg = extra['STAGE2']
		num_channels = self.stage2_cfg['NUM_CHANNELS']
		num_channels = [num_channels[i] for i in range(len(num_channels))]
		self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
		self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

		self.stage3_cfg = extra['STAGE3']
		num_channels = self.stage3_cfg['NUM_CHANNELS']
		num_channels = [num_channels[i] for i in range(len(num_channels))]
		self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
		self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

		self.stage4_cfg = extra['STAGE4']
		num_channels = self.stage4_cfg['NUM_CHANNELS']
		num_channels = [num_channels[i] for i in range(len(num_channels))]
		self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
		self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)
		
		# print(pre_stage_channels)
		last_inp_channels = np.int(np.sum(pre_stage_channels))
		self.last_layer = OutSigmoid(last_inp_channels)
		self.projection = MlpNorm(pre_stage_channels[0])

	def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
		num_branches_cur = len(num_channels_cur_layer)
		num_branches_pre = len(num_channels_pre_layer)

		transition_layers = []
		for i in range(num_branches_cur):
			if i < num_branches_pre:
				if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
					transition_layers.append(nn.Sequential(
						nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
						nn.BatchNorm2d(num_channels_cur_layer[i]),
						nn.PReLU()))
				else:
					transition_layers.append(None)
			else:
				conv3x3s = []
				for j in range(i+1-num_branches_pre):
					inchannels = num_channels_pre_layer[-1]
					outchannels = num_channels_cur_layer[i] if j == i-num_branches_pre else inchannels
					conv3x3s.append(nn.Sequential(
						nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
						nn.BatchNorm2d(outchannels),
						nn.PReLU()))
				transition_layers.append(nn.Sequential(*conv3x3s))
		return nn.ModuleList(transition_layers)

	def _make_layer(self, in_c, out_c, blocks, stride=1):
		downsample = None
		if stride != 1 or in_c != out_c:
			downsample = nn.Sequential(
				nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(out_c),
			)

		layers = []
		layers.append(Bottleneck(in_c, out_c, stride, downsample))
		in_c = out_c
		for i in range(1, blocks):
			layers.append(Bottleneck(in_c, out_c))
		return nn.Sequential(*layers)

	def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
		num_modules = layer_config['NUM_MODULES']
		num_branches = layer_config['NUM_BRANCHES']
		num_blocks = layer_config['NUM_BLOCKS']#[3,]*num_branches#
		num_channels = layer_config['NUM_CHANNELS']

		modules = []
		for i in range(num_modules):
			# multi_scale_output is only used last module
			if not multi_scale_output and i == num_modules - 1:
				reset_multi_scale_output = False
			else:
				reset_multi_scale_output = True
			# print('HRModule')
			modules.append(
				HRModule(num_branches, num_blocks,
									  num_inchannels, num_channels,
									  reset_multi_scale_output)
			)
			num_inchannels = modules[-1].num_inchannels
		return nn.Sequential(*modules), num_inchannels

	def regular(self, sampler, lab, fov=None, return_loss=True):
		emb = sampler.select(self.feat, self.pred.detach(), lab, fov)
		# print(emb.shape)
		emb = self.projection(emb)
		# print(emb.shape)
		self.emb = emb
		if return_loss:
			return sampler.infonce(emb)

	def forward(self, x, mask=None, temp=1):
		x = self.layer0(x)
		x = self.layer1(x)

		x_list = []
		for i in range(self.stage2_cfg['NUM_BRANCHES']):
			if self.transition1[i] is not None:
				x_list.append(self.transition1[i](x))
			else:
				x_list.append(x)
		y_list = self.stage2(x_list)

		x_list = []
		for i in range(self.stage3_cfg['NUM_BRANCHES']):
			if self.transition2[i] is not None:
				x_list.append(self.transition2[i](y_list[-1]))
			else:
				x_list.append(y_list[i])
		y_list = self.stage3(x_list)
		self.feat= y_list[0]

		x_list = []
		for i in range(self.stage4_cfg['NUM_BRANCHES']):
			if self.transition3[i] is not None:
				x_list.append(self.transition3[i](y_list[-1]))
			else:
				x_list.append(y_list[i])
		x = self.stage4(x_list)
		# print(x[0].shape, x[1].shape, x[2].shape, x[3].shape)

		x0_h, x0_w = x[0].size(2), x[0].size(3)
		x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
		x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
		x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
		# print(x[0].shape, x1.shape, x2.shape, x3.shape)
		
		# self.feat = F.normalize(x[0], p=2, dim=1)

		pred = torch.cat([x[0], x1, x2, x3], 1)
		self.pred = self.last_layer(pred)
		return self.pred

# high_resoluton_net related params for segmentation
from yacs.config import CfgNode as CN
HRNET = CN()
NUM_CELLS = 3
NUM_MODULES = 1
HRNET.STAGE1 = CN()
HRNET.STAGE1.NUM_MODULES = NUM_MODULES
HRNET.STAGE1.NUM_BRANCHES = 1
HRNET.STAGE1.NUM_BLOCKS = [NUM_CELLS]
HRNET.STAGE1.NUM_CHANNELS = [16]

HRNET.STAGE2 = CN()
HRNET.STAGE2.NUM_MODULES = NUM_MODULES
HRNET.STAGE2.NUM_BRANCHES = 2
HRNET.STAGE2.NUM_BLOCKS = [NUM_CELLS, ]*2
HRNET.STAGE2.NUM_CHANNELS = [16, 16]

HRNET.STAGE3 = CN()
HRNET.STAGE3.NUM_MODULES = NUM_MODULES
HRNET.STAGE3.NUM_BRANCHES = 3
HRNET.STAGE3.NUM_BLOCKS = [NUM_CELLS, ]*3
HRNET.STAGE3.NUM_CHANNELS = [16, 16, 16]

HRNET.STAGE4 = CN()
HRNET.STAGE4.NUM_MODULES = NUM_MODULES
HRNET.STAGE4.NUM_BRANCHES = 4  
HRNET.STAGE4.NUM_BLOCKS = [NUM_CELLS, ]*4
HRNET.STAGE4.NUM_CHANNELS = [16, 16, 16, 16]
# 32, 64, 96, 128

def hrba(in_channels=1, **args):
	HRNet.MyConv = BasicConv2d
	return HRNet(HRNET, inp_planes=in_channels, name='hrba', **args)
def hrpy(in_channels=1, **args):
	HRNet.MyConv = PyridConv2d
	return HRNet(HRNET, inp_planes=in_channels, name='hrpy', **args)
def hrsp(in_channels=1, **args):
	HRNet.MyConv = SpAttConv2d
	return HRNet(HRNET, inp_planes=in_channels, name='hrsp', **args)
def hrsc(in_channels=1, **args):
	HRNet.MyConv = SCNetConv2d
	return HRNet(HRNET, inp_planes=in_channels, name='hrsc', **args)
def hrdo(in_channels=1, **args):
	HRNet.MyConv = DoverConv2d
	return HRNet(HRNET, inp_planes=in_channels, name='hrdo', **args)
#end#
# from attention import *
'''

_base_ = './ocrnet_hr18_512x512_20k_voc12aug.py'
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w18_small',
    backbone=dict(
        extra=dict(
            stage1=dict(num_blocks=(2, )),
            stage2=dict(num_blocks=(2, 2)),
            stage3=dict(num_modules=3, num_blocks=(2, 2, 2)),
            stage4=dict(num_modules=2, num_blocks=(2, 2, 2, 2)))))
			'''
			

		# feats = mlp_sample_selection(feats, prob.detach(), mask, mode='hard')


if __name__ == '__main__':
	net = hrdo()


	x = torch.rand(2,1,64,64)

	net.eval()
	ys = net(x)

	for y in ys:
		print(y.shape)
	# print(net.__name__, y['loss'])

	# net.train()
	# sampler = MLPSampler(top=4, low=0, mode='hard')
	# l = net.regular(sampler, torch.rand(2,1,64,64), torch.rand(2,1,64,64), False)
	# print(net.__name__, l.item())


	# plot(net.emb)

	print('Params model:',sum(p.numel() for p in net.parameters() if p.requires_grad))
