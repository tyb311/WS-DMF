
# Loosely inspired on https://github.com/jvanvugt/pytorch-unet
# Improvements (conv_bridge, shortcut) added by A. Galdran (Dec. 2019)


import kornia
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

#start#
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

DIVIDES=12
def kernelsMatch(ksize=11, sigma=1.5, ylen=10):#1,10
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

class MatchUnitConst(nn.Module):
	def __init__(self, kernel=np.random.rand(9,9)):#13,17
		super(MatchUnitConst, self).__init__()
		self.padding = kernel.shape[0]//2
		kernel = torch.from_numpy(kernel).type(torch.float32).unsqueeze(0).unsqueeze(0)
		self.weight = nn.Parameter(kernel, requires_grad=False)
	def forward(self, x):
		return F.conv2d(x, self.weight, bias=None, padding=self.padding)

class MatchUnitDynamic(nn.Module):
	def __init__(self, kernel=np.random.rand(9,9)):#13,17
		super(MatchUnitDynamic, self).__init__()
		self.padding = kernel.shape[0]//2
		kernel = torch.from_numpy(kernel).type(torch.float32).unsqueeze(0).unsqueeze(0)
		# kernel = torch.rand_like(kernel)
		self.weight = nn.Parameter(kernel, requires_grad=True)
		# self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)
	def forward(self, x):
		return F.conv2d(x, self.weight, bias=None, padding=self.padding)

class MatchGroup(nn.Module):
	def __init__(self, ksize=11, sigma=1.5, match_unit=MatchUnitConst):#13,17
		super(MatchGroup, self).__init__()
		# self.out = nn.Conv2d(12,1,1,1,0)
		kernels = kernelsMatch(ksize, sigma)
		self.matches = nn.ModuleList([match_unit(kernel) for kernel in kernels])

	def forward(self, x):
		outputs = [self.matches[i](x) for i in range(self.matches.__len__())]
		outputs = torch.cat(outputs, dim=1)
		outputs = torch.max(outputs, dim=1)[0].unsqueeze(1)
		# print('MatchGroup:', outputs.shape)
		return outputs
		# return self.out(outputs)

class MatchMS(nn.Module):# Multi-Scale Match
	def __init__(self, ksizes=[5,7,9,11], sigmas=[1.5,1.5,1.5,1.5], dynamic=False):
		super(MatchMS, self).__init__()
		self.channels = len(ksizes)
		self.out = nn.Sequential(
				nn.Conv2d(1,1,kernel_size=1,stride=1,padding=0,bias=True), nn.Sigmoid()
				) if dynamic else nn.Sequential()
		self.kernels = nn.ModuleList()
		match_unit = MatchUnitDynamic if dynamic else MatchUnitConst
		for ksize, sigma in zip(ksizes, sigmas):
			generator = MatchGroup(ksize=ksize, sigma=sigma, match_unit=match_unit)
			# self.add_module("gen_" + str(ksize), generator)
			self.kernels.append(generator)

	def forward(self, x):
		outputs = torch.cat([generator(x) for generator in self.kernels], dim=1)
		# a,b = torch.max(outputs, dim=1)
		# print('MAX:', outputs.shape, a.shape, b.shape)
		outputs = torch.max(outputs, dim=1)[0].unsqueeze(1)
		return self.out(outputs)
		# return outputs#self.out(outputs)

class DMF(torch.nn.Module):	## neural matched-filtering attention net.  DMF Net
	__name__ = 'dmf'
	def __init__(self, res=True, filters=32):
		super(DMF, self).__init__()
		n_classes=1
		phase = 2

		self.loss_preception = PerceptionLoss()
		self.loss_texture = TextureLoss()

		self.mf_learn = MatchMS(dynamic=True)
		self.mf_const = MatchMS(dynamic=False)

		# print('Channels=', channels)
		if res:
			self.net_seg = LittleRes(in_c=phase, n_classes=n_classes)
			name_seg = 'Res'
		else:
			# self.net_seg = LittleUNet(in_c=phase, n_classes=n_classes, layers=(32,32,32,32), out=nn.Sigmoid())
			# self.net_seg = LittleUNet(in_c=phase, n_classes=n_classes, layers=(16,16,16,16), out=nn.Sigmoid())
			self.net_seg = LUNet(phase, n_classes=n_classes, layers=(filters,)*4)
			name_seg = self.net_seg.__name__
			
		self.projector = self.net_seg.projector
		self.__name__ += name_seg
		# print(self.__name__)

	def constraint(self, lab=None, fov=None, **args):
		# 算法的效果好不好，应该结合主观和客观来评价，下图的客观数据也表明：MS-SSIM+L1损失函数是最好的
		l_mse = F.mse_loss(self.tmp['learn'], self.tmp['const'])
		# l_mae = F.l1_loss(self.tmp['learn'], self.tmp['const'])
		# print('L1 & L2:', l_mse.item(), l_mae.item())
		# ssim = kornia.ssim(self.tmp['learn'], self.tmp['const'], window_size=9, reduction='mean')
		# l_ssim = (1 - ssim)/2.0
		# print('L1 & SSIM:', l_mae.item(), l_ssim.item())
		# l_psnr = kornia.losses.psnr_loss(self.tmp['learn'], self.tmp['const'], max_val=2.0)
		# print('Regular:', l_mse.item(), l_ssim.item(), l_psnr.item())
		# return l_mae + l_mse# + l_ssim*0.1##
		# return l_mae
		l_per = self.loss_preception(self.tmp['learn'], self.tmp['const'])*0.1
		return l_mse + l_per

	def regular(self, sampler, lab, fov=None, return_loss=True):
		emb = sampler.select(self.feat, self.pred.detach(), lab, fov)
		# print(emb.shape)
		emb = self.projector(emb)
		# print(emb.shape)
		self.emb = emb
		if return_loss:
			return sampler.infonce(emb)

	tmp = {}
	def forward(self, x):
		c = self.mf_const(x)
		self.tmp['const'] = c

		l = self.mf_learn(x)
		self.tmp['learn'] = l

		# print(x.shape, c.shape, l.shape)
		f = torch.cat([x, l], dim=1)

		o = self.net_seg(f)
		self.feat = self.net_seg.feat
		self.pred = o
		# print(y.shape)
		# self.tmp['out'] = o
		return o

def dmf32():	#deep matched filtering
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
	x = torch.rand(2,1,128,256)
	# net = LittleUNet()
	net = dmf32()
	# net = dmfu32()
	print(net.__name__)
	
	# print('mf_learn:',sum(p.numel() for p in net.mf_learn.parameters() if p.requires_grad))
	# print('net_seg:',sum(p.numel() for p in net.net_seg.parameters() if p.requires_grad))
	# print('Params total:',sum(p.numel() for p in net.parameters() if p.requires_grad))

	# net.eval()
	import time
	st = time.time()
	ys = net(x)
	print('Time:', time.time() - st)


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

	losAux = net.constraint(gt=x)*0.01
	print('Params model:',sum(p.numel() for p in net.parameters() if p.requires_grad))