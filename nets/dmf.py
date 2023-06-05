
# Loosely inspired on https://github.com/jvanvugt/pytorch-unet
# Improvements (conv_bridge, shortcut) added by A. Galdran (Dec. 2019)


import time
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

import sys
sys.path.append('.')
sys.path.append('..')

# from utils.sample import *
from nets import *
from utils import *

class LittleRes(nn.Module):
	iterations=14
	def __init__(self, in_c, n_classes, stride=1, out='dis', **args):
		super(LittleRes, self).__init__()
		self.convs = nn.ModuleList([
			nn.Sequential(
				nn.Conv2d(in_c, in_c, kernel_size=3, stride=1, padding=1, bias=True),
				nn.LeakyReLU(),
			) for i in range(self.iterations)
		])
		self.final = nn.Sequential(
				nn.Conv2d(in_c,1,kernel_size=3,stride=1,padding=1,bias=True), 
				nn.Sigmoid()
				)
	def forward(self, x):
		for i in range(self.iterations):
			x = self.convs[i](x) + x
		return self.final(x)

#start#
DIVIDES=12
# def kernelsMatch(ksize=13, sigma=1.5, ylens=range(3,8)):#1,10
def kernelsMatch(ksize=11, sigma=1.5, ylen=5):#1,10
	halfLength = ksize // 2
	sqrt2pi_sigma = np.sqrt(2 * np.pi) * sigma
	x, y = np.meshgrid(range(ksize), range(ksize))
	x -= halfLength
	y -= halfLength
	kernels = []
	# for ylen in ylens:
	for theta in np.arange(0, np.pi, np.pi/DIVIDES):
		cos, sin = np.cos(theta), np.sin(theta)
		x_ = x * cos + y * sin
		y_ = y * cos - x * sin 

		indexZero = np.logical_or(abs(x_) > 3*sigma, abs(y_) > ylen)
		kernel = -np.exp(-0.5 * (x_ / sigma) ** 2) / sqrt2pi_sigma
		kernel[indexZero] = 0
	
		indexFalse = kernel<0
		mean = np.sum(kernel) / np.sum(indexFalse)
		kernel[indexFalse] -= mean
		# print('Kernel:', kernel.min(), kernel.max())
		kernels.append(kernel)
	return kernels
# KERNELS = kernelsMatch()
# print(len(KERNELS), KERNELS[0].shape)

class MatchUnit(nn.Module):
	def __init__(self, kernel=np.random.rand(9,9)):#13,17
		super(MatchUnit, self).__init__()
		self.padding = kernel.shape[0]//2
		kernel = torch.from_numpy(kernel).type(torch.float32).unsqueeze(0).unsqueeze(0)
		# kernel = torch.rand_like(kernel)
		self.constant = nn.Parameter(kernel, requires_grad=False)
		self.dynamics = nn.Parameter(kernel, requires_grad=True)
		self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)
	def regular(self):
		return F.l1_loss(self.constant, self.dynamics) + F.mse_loss(self.constant, self.dynamics)
	def forward_con(self, x):
		with torch.no_grad():
			return F.conv2d(x, self.constant, bias=None, padding=self.padding)
	def forward_dyn(self, x):
		return F.conv2d(x, self.dynamics, bias=self.bias, padding=self.padding)

loss_preception = PerceptionLoss()
# loss_texture = TextureLoss()
class MatchGroup(nn.Module):
	def __init__(self, ksize=11, sigma=1.5, ylen=5):#13,17
		super(MatchGroup, self).__init__()
		# self.out = nn.Conv2d(12,1,1,1,0)
		kernels = kernelsMatch(ksize, sigma, ylen)
		self.units = nn.ModuleList([MatchUnit(kernel) for kernel in kernels])
		self.max = nn.Sequential(
				nn.Conv2d(12,1,kernel_size=1,stride=1,padding=0,bias=True), nn.BatchNorm2d(1),nn.Sigmoid()
				)

	def regular(self):
		return self.loss_reg + self.loss_per + sum(u.regular() for u in self.units)
	def forward(self, x):
		f_con = torch.cat([m.forward_con(x) for m in self.units], dim=1)
		r_con = torch.max(f_con, dim=1)[0].unsqueeze(1)

		f_dyn = torch.cat([m.forward_dyn(x) for m in self.units], dim=1)
		r_dyn = self.max(f_dyn)#[0].unsqueeze(1)
		# print('MatchGroup:', outputs.shape)
		self.loss_reg = F.l1_loss(f_con, f_dyn) + F.mse_loss(f_con, f_dyn)
		if loss_preception.device != x.device:
			loss_preception = loss_preception.to(x.device)
		self.loss_per = loss_preception(r_con, r_dyn)
		return r_dyn

class MatchMS(nn.Module):# Multi-Scale Match
	def __init__(self, ksizes=[7,9,11], sigmas=[1.5,1.5,1.5,1.5,1.5], ylens=[3,4,5,6,7]):
		super(MatchMS, self).__init__()
		self.channels = len(ksizes)
		self.groups = nn.ModuleList()
		for ksize, sigma, ylen in zip(ksizes, sigmas, ylens):
			group = MatchGroup(ksize=ksize, sigma=sigma, ylen=ylen)
			self.groups.append(group)
	def regular(self):
		return sum([g.regular() for g in self.groups])
	def forward(self, x):
		outputs = torch.cat([group(x) for group in self.groups], dim=1)
		return outputs#self.out(outputs)

class DMF(torch.nn.Module):	## neural matched-filtering attention net.  DMF Net
	__name__ = 'dmf'
	def __init__(self, res=True, n_classes=1, filters=32):
		super(DMF, self).__init__()

		self.net_dmf = MatchMS()
		channels = self.net_dmf.channels
		# print('Channels=', channels)

		if res:
			self.net_seg = LittleRes(in_c=channels, n_classes=n_classes)
		else:
			self.net_seg = LUNet(channels, n_classes=n_classes, layers=(filters,)*4)
		self.__name__ += self.net_seg.__name__
		# print(self.__name__)

	def regular(self, **args):
		return self.net_dmf.regular()

	tmp = {}
	def forward(self, x):
		# st = time.time()
		f = self.net_dmf(x)
		self.tmp['learn'] = f
		# print(f.shape)
		# print('Time for DMF:', time.time()-st)

		# st = time.time()
		o = self.net_seg(f)
		# print('Time for SEG:', time.time()-st)
		self.feat = self.net_seg.feat
		self.pred = o
		# print(y.shape)
		# self.tmp['out'] = o
		return o

def dmf32(**args):	#deep matched filtering
	net = DMF(res=False, filters=32)
	net.__name__ = 'dmf32'
	return net
def dmf16():	#deep matched filtering
	net = DMF(res=False, filters=16)
	net.__name__ = 'dmf16'
	return net
#end#


'''

represent layer的约束，罗师兄的attention损失思想就很好，既保证单层响应最大化，又保证其他方向响应最小化就可以了。
又相比于dris的手工高斯血管更符合实际，而且血管符号和他的高斯是相反的
'''




# G:\Objects\HisEye\EyeExp05a\0517drive-dmf32L2P1-fr没想到我也有这么好的dmf实验结果哈哈
if __name__ == '__main__':

	# x = torch.rand(2,1,512,512)
	x = torch.rand(4,1,64,64)
	# net = LittleUNet()
	net = dmf32()
	# net = dmfu32()
	
	# print('net_dmf:',sum(p.numel() for p in net.net_dmf.parameters() if p.requires_grad))
	# print('net_seg:',sum(p.numel() for p in net.net_seg.parameters() if p.requires_grad))
	# print('Params total:',sum(p.numel() for p in net.parameters() if p.requires_grad))

	# net.eval()
	import time
	st = time.time()
	ys = net(x)
	print(net.__name__, net.feat.shape)
	print('Time:', time.time() - st)
	print('regular:', net.regular())


	# st = time.time()
	# # print('Regular:', net.__name__, net.constraint().item())
	# print('Regular:', net.__name__)
	# net.constraint(x.round())
	# print('Time:', time.time() - st)

	if isinstance(ys, list):
		for y in ys:
			print(y.shape, y.min().item(), y.max().item())
	else:
		print(ys.shape, ys.min().item(), ys.max().item())

		

	# feats = MlpNorm.sample(net.feats, x.detach(), x.round())
	# emb = net.projection(feats)
	# # emb = mlp_sample_selection_by_rank(self.feats, prob, mask)
	# MlpNorm.plot(emb)
	# plt.show()

	print('Params model:',sum(p.numel() for p in net.parameters() if p.requires_grad))