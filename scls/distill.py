# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


import sys
sys.path.append('.')
sys.path.append('..')

from utils import *
from nets import *
from scls import *

#start#
class Distill(nn.Module):
	__name__ = 'distill'
	def __init__(self, encoder, **kwargs):
		super().__init__()

		self.encoder = encoder
		# self.__name__ = self.encoder.__name__
		self.__name__ = 'X'.join([self.__name__, self.encoder.__name__]) #, clloss

	def forward(self, img, **args):#只需保证两组数据提取出的特征标签顺序一致即可
		out = self.encoder(img, **args)
		self.pred = self.encoder.pred
		self.feat = self.encoder.feat
		# self._dequeue_and_enqueue(proj1_ng, proj2_ng)
		if hasattr(self.encoder, 'tmp'):
			self.tmp = self.encoder.tmp
		return out

	def constraint(self, **args):
		return self.encoder.constraint(**args)

	def regular(self, sampler, lab, fov=None):#contrastive loss split by classification
		feat = sampler.select(self.feat.clone(), self.pred.detach(), lab, fov)
		# print(emb.shape)
		proj = self.projector(feat)
		self.proj = proj
		pred = self.predictor(proj)
		# random mapping
		# pred, proj = sampler.norm(pred, proj)
		# rand = torch.randn(64, 64, device=feat.device)
		# pred = F.normalize(pred @ rand, dim=-1)
		# proj = F.normalize(proj @ rand, dim=-1)

		# compute loss
		losSG1 = self.loss(pred, proj.detach(), temperature=self.temperature)
		# losSG2 = self.loss(proj, pred.detach(), temperature=self.temperature)

		# losAU = align_uniform(pred, proj.detach()) 
		# fh, fl, bh, bl = torch.chunk(proj, chunks=4, dim=0)
		# losL2 = F.mse_loss(fh, fl) + F.mse_loss(bh, bl)
		# losAU = align_loss(fh, fl) + align_loss(bh, bl)
		# losKL = compute_kl_loss(fh, fl) + compute_kl_loss(bh, bl)
		return losSG1# + losL2# + losAU# + losKL#
#end#    


import cv2
if __name__ == '__main__':
	pred = cv2.imread('figures/z_pred.png', cv2.IMREAD_GRAYSCALE)[:512, :512]
	mask = cv2.imread('figures/z_mask.png', cv2.IMREAD_GRAYSCALE)[:512, :512]
	true = cv2.imread('figures/z_true.png', cv2.IMREAD_GRAYSCALE)[:512, :512]
	h, w = pred.shape

	pred = torch.from_numpy(pred.astype(np.float32)/255).unsqueeze(0).unsqueeze(0)
	mask = torch.from_numpy(mask.astype(np.float32)/255).unsqueeze(0).unsqueeze(0)
	true = torch.from_numpy(true.astype(np.float32)/255).unsqueeze(0).unsqueeze(0)
	print('imread:', pred.shape, mask.shape, true.shape)

	feat = torch.rand(1,32,h,w)
	feat = F.normalize(feat, p=2, dim=1)

	

	sampler = MLPSampler(top=4, low=0, mode='prob')


	# net = Distill(encoder=lunet(), clloss='sim2')
	net = Distill(encoder=lunet(), clloss='sim2', proj_num_length=32)
	# net.eval()
	ys = net(torch.rand_like(pred))

	for y in ys:
		print(y.shape)
	# print(net.__name__, y['loss'])

	# # net.train()
	# l = net.regular(sampler, pred, mask)
	# # l = net.regular3(sampler, pred, mask)
	# print(net.__name__, l.item())


	# l = net.sphere(sampler, pred, mask)
	# print(net.__name__, l.item())
	# plot(net.emb)