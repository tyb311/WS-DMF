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
		# https://blog.csdn.net/bxdzyhx/article/details/112729725
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
		# losSTD = torch.clamp_min(-raw.std(),-0.5)*8+4#torch.abs(raw.std()-1)
		losSTD = torch.clamp_min(-raw.reshape(raw.shape[0], raw.shape[1], -1).std(dim=-1).mean(),-0.5)*8+4
		# losSTD = torch.clamp_min(-torch.norm(raw),-0.5)*1+4#torch.abs(raw.std()-1)

		# 随机找个核，做旋转正交损失
		#############################################
		# # print('kernel:', raw.shape)
		# kernel1 = raw[0:1,0:1]
		# # for i in range(1, self.channels):
		# i = random.randint(1, self.channels)
		# kernel2 = raw[i:i+1,i:i+1]
		# grid = eval('self.grid'+str((i+self.chs_half)%self.channels))
		# kernel3 = F.grid_sample(kernel1, grid, align_corners=True)
		# resCST = torch.norm(kernel2*kernel2)#.sum()
		# resOth = torch.norm(kernel2*kernel3)#.sum()
		# losRand = resOth / (resCST+resOth+1e-6)#参考dice的形式
		# return losRand + losSTD

		# 随机找个角度，做旋转正交损失
		#############################################
		# angle = random.randint(1, self.channels-1)
		# index = (self.chs_half+angle)%self.channels
		# dis = torch.cat([raw[:,index:], raw[:,:index]], dim=1)

		# grid = eval('self.grids'+str(angle))
		# oth = F.grid_sample(raw, grid, align_corners=True)

		# cst = F.grid_sample(raw, self.wgt_cst, align_corners=True)
		# resCST = torch.norm(raw*cst)#.sum()
		# resOth = torch.norm(oth*dis)#.sum()
		# losOth = resOth / (resCST+resOth+1e-6)#参考dice的形式
		# # losOth = resOth / (resCST+1e-6)#参考dice的形式
		# # losOth = -torch.log(1-resOth / (resCST+resOth+1e-6))
		# return losOth + losSTD

		
		# 固定找个角度，做旋转正交损失
		#############################################
		dis = torch.chunk(raw, chunks=2, dim=1)
		# print(dis[1].shape, dis[0].shape)
		dis = torch.cat([dis[1], dis[0]], dim=1)
		# print(raw.shape, dis.shape)

		# centrosymmetric，旋转180度与原本一致，是为中心对称
		cst = F.grid_sample(raw, self.wgt_cst, align_corners=True)
		# orthometric，旋转90度与对应模板一致
		# oth = F.grid_sample(raw, self.wgt_oth, align_corners=True)
		# losCST = F.mse_loss(cst, raw.detach()) + F.mse_loss(dis.detach(), oth)
		# return losCST + losSTD

		# print(raw.shape, cst.shape, dis.shape)
		# resCST = torch.norm(F.conv2d(raw, cst, bias=None, padding=0, groups=oth.shape[1]))
		# resOth = torch.norm(F.conv2d(raw, dis, bias=None, padding=0, groups=oth.shape[1]))
		resCST = torch.norm(raw*cst)#.sum()
		resOth = torch.norm(raw*dis)#.sum()
		losOth = resOth / (resCST+resOth+1e-6)#参考dice的形式
		# losOth = resOth / (resCST+1e-6)#参考dice的形式
		# losOth = -torch.log(1-resOth / (resCST+resOth+1e-6))
		return losOth + losSTD
			
		# 可视化
		#############################################
		# raw = make_grid(raw[0:1].permute(1,0,2,3), nrow=4)
		# dis = make_grid(dis[0:1].permute(1,0,2,3), nrow=4)
		
		# raw = make_grid(raw[0:1].permute(1,0,2,3), nrow=4)
		# dis = make_grid(cst[0:1].permute(1,0,2,3), nrow=4)
		
		# raw = make_grid(raw[0:1].permute(1,0,2,3), nrow=4)
		# dis = make_grid(oth[0:1].permute(1,0,2,3), nrow=4)

		# plt.subplot(121),plt.imshow(raw.permute(1,2,0).data.numpy())
		# plt.subplot(122),plt.imshow(dis.permute(1,2,0).data.numpy())
		# plt.show()

		# print(losCST.item(), resCST.item(), resOth.item(), losOth.item())
		# return losOth + losCST + losSTD
#end#

def plot(ret):
	title = '[{:.4f}@{:.4f}->{:.4f}]'.format(ret.std().item(), ret.min().item(), ret.max().item())
	ret = ret - ret.min()
	ret = ret / ret.max()
	ret = make_grid(ret[0:1].permute(1,0,2,3), nrow=4)
	ret = ret.permute(1,2,0).data.numpy().astype(np.float32)
	plt.imshow(ret)
	plt.title(title)

if __name__ == '__main__':
	idx_rot = 2
	net = nn.Sequential(
		nn.Conv2d(1,16,3,1,1),
		nn.BatchNorm2d(16),
		nn.Conv2d(16,16,3,1,1),
		nn.Conv2d(16,16,3,1,1),
		nn.Conv2d(16,16,3,1,1),
		nn.Conv2d(16,1,3,1,1),
		nn.Sigmoid()
	)
	# net[idx_rot].weight = nn.Parameter(torch.rand(16,16,3,3)*0.8-0.5, requires_grad=True)
	print(net)
	net.train()

	rot = Rotation()
	# los = rot(raw)
	# print('Los:', los.item())


	# dis = torch.chunk(raw, chunks=2, dim=1)
	# dis = torch.cat([dis[1], dis[0]], dim=1)
	# raw = make_grid(raw[0:1].permute(1,0,2,3), nrow=4)
	# dis = make_grid(dis[0:1].permute(1,0,2,3), nrow=4)
	# plt.subplot(121),plt.imshow(raw.permute(1,2,0).data.numpy())
	# plt.subplot(122),plt.imshow(dis.permute(1,2,0).data.numpy())
	# plt.show()

	# x = torch.rand(2,16,64,64)#-0.5
	# y = torch.rand(2, 1,64,64).round()
	path_rgb = r'G:\Objects\datasets\seteye\eyeraw\chase\test_rgb/21_test.jpg'
	path_lab = r'G:\Objects\datasets\seteye\eyeraw\chase\test_lab/21_manual1.png'
	x = cv2.imread(path_rgb, 0)[300:428,300:428]
	y = cv2.imread(path_lab, 0)[300:428,300:428]
	x = torch.from_numpy(x).reshape(1,1,128,128).float()/255
	y = torch.from_numpy(y).reshape(1,1,128,128).float()/255
	# plt.subplot(121),plt.imshow(x.squeeze().data.numpy())
	# plt.subplot(122),plt.imshow(y.squeeze().data.numpy())
	# plt.show()

	optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999), weight_decay=2e-4)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,  
		mode='min', factor=0.7, patience=2, 
		verbose=False, threshold=0.0001, threshold_mode='rel', 
		cooldown=2, min_lr=1e-5, eps=1e-9)

	plt.subplot(121),plot(net[idx_rot].weight)
	for i in range(256):
		optimizer.zero_grad()
		# p = F.conv2d(x, raw, padding=1)
		p = net(x)

		losSEG = F.mse_loss(p, y)

		losROT = rot(net[idx_rot].weight)*1
		print(i, losSEG.item(), losROT.item())
		# print(i, los.item(), los1.item(), los.item())

		los = losSEG + losROT
		los.backward()
		optimizer.step()
		scheduler.step(los.item())

	raw = net[idx_rot].weight
	plt.subplot(122),plot(net[idx_rot].weight)
	plt.show()