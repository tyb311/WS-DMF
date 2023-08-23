import time
import math, random
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchvision.utils import make_grid

#start#
class Rotation(nn.Module):
	def __init__(self, channels=16, ksize=3):#180划分16份就是每次旋转10度
		super(Rotation, self).__init__()
		idx_oth = channels//2
		idx_cst = channels-1
		self.channels = channels
		self.chs_half = channels//2
		for i,theta in enumerate(torch.arange(0, np.pi, np.pi/channels)):
			grid = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0], [torch.sin(theta), torch.cos(theta), 0]])
			grid = F.affine_grid(grid[None, ...], (1,1,ksize,ksize), align_corners=True)
			self.register_buffer('grid'+str(i), grid.type(torch.float32))
			self.register_buffer('grids'+str(i), grid.type(torch.float32).repeat(channels,1,1,1))
			if i==idx_oth:
				self.register_buffer('wgt_oth', grid.type(torch.float32).repeat(channels,1,1,1))
			elif i==idx_cst:
				self.register_buffer('wgt_cst', grid.type(torch.float32).repeat(channels,1,1,1))
	
	def initialize(self, raw, channels=16):
		for i,theta in enumerate(torch.arange(0, np.pi, np.pi/channels)):
			grid = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0], [torch.sin(theta), torch.cos(theta), 0]])
			grid = F.affine_grid(grid[None, ...], (1,1,ksize,ksize), align_corners=True)

	def forward(self, raw):
		losSTD = torch.clamp_min(-raw.reshape(raw.shape[0], raw.shape[1], -1).std(dim=-1).mean(),-0.5)*8+4


		# 固定找个角度，做旋转正交损失
		#############################################
		dis = torch.chunk(raw, chunks=2, dim=1)
		# print(dis[1].shape, dis[0].shape)
		dis = torch.cat([dis[1], dis[0]], dim=1)
		# print(raw.shape, dis.shape)

		# centrosymmetric，旋转180度与原本一致，是为中心对称
		cst = F.grid_sample(raw, self.wgt_cst, align_corners=True)

		resCST = torch.norm(raw*cst)#.sum()
		resOth = torch.norm(raw*dis)#.sum()
		losOth = resOth / (resCST+resOth+1e-6)#参考dice的形式
		# losOth = resOth / (resCST+1e-6)#参考dice的形式
		# losOth = -torch.log(1-resOth / (resCST+resOth+1e-6))
		return losOth + losSTD
			
#end#

