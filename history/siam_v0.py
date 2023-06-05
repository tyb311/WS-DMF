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

from utils.sample import *
from attention import *
from dou import *

# def infonce_matrix2(q, k, temperature=0.1):
# 	# positive logits: Nx1
# 	qf, qb = torch.chunk(q, 2)
# 	kf, kb = torch.chunk(k, 2)

# 	l_pos = torch.einsum('nc,kc->nk', [qf, kf])
# 	l_pos = torch.exp(l_pos / temperature)
# 	l_neg = torch.einsum('nc,kc->nk', [qf, kb])
# 	l_neg = torch.exp(l_neg / temperature).sum(dim=1, keepdim=True)
# 	los1 = - torch.log(l_pos / (l_pos + l_neg)).mean()

# 	l_pos =  torch.einsum('nc,kc->nk', [qb, kb])
# 	l_pos = torch.exp(l_pos / temperature)
# 	l_neg =  torch.einsum('nc,kc->nk', [qb, kf])
# 	l_neg = torch.exp(l_neg / temperature).sum(dim=1, keepdim=True)
# 	los2 = - torch.log(l_pos / (l_pos + l_neg)).mean()
# 	return los1# + los2

#start#
def align_uniform(q, k, temperature=0.1):
	# “Understanding contrastive representation learning through alignment and uniformity on the hypersphere,” ICML 2020
	#https://github.com/SsnL/align_uniform
	def align_loss(x, y, alpha=2):
		assert x.shape==y.shape, "shape of postive sample must be the same"
		return (x - y).norm(p=2, dim=1).pow(alpha).mean()

	def uniform_loss(x, t=2):
		return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
		#pdist If input has shape N×M then the output will have shape 1/2 * N * (N - 1) 

	return align_loss(q, k) + uniform_loss(q)

def infonce_matrix2(q, k, temperature=0.1):
	"""
	“Contrastive Semi-Supervised Learning for 2D Medical Image Segmentation,” arXiv:2106.06801 [cs]
	true is one hot class vector of patches (S, num_classes)
	feat = (S, dim)
	"""
	true = torch.cat([
		torch.ones(size=(q.shape[0]//2, 1), dtype=torch.float32),
		torch.zeros(size=(q.shape[0]//2, 1), dtype=torch.float32)
	], dim=0).to(q.device)
	# idx_pos[i][j] = (class[i] == class[j])
	idx_pos = torch.mm(true, true.permute(1,0))

	# sim_mat[i][j] = cosine angle between ith and jth embed
	sim_mat = torch.mm(q, k.permute(1,0)) / temperature
	
	#########################################################################################
	# For stability
	#########################################################################################
	sim_max, _ = torch.max(sim_mat, dim=1, keepdim=True)
	sim_mat = sim_mat - sim_max.detach()
	# print(sim_mat.min(), sim_mat.max())

	sim_exp = torch.exp(sim_mat)
	# sim_negs = sum or exp(similarity) of ith patch with every negative patch

	#########################################################################################
	# The hard negative sample z = a*zi + (1 - a)*zn. sim(z, zi) = a + (1 - a)*sim(zn, zi)
	#########################################################################################
	# alpha = torch.rand(1, dtype=torch.float32)*0.4
	# alpha = np.random.rand()*0.4
	# sim_mix = torch.exp(alpha + (1. - alpha) * sim_mat)
	sim_neg = sim_exp# + sim_mix
	sim_neg = torch.sum(sim_exp * (1 - idx_pos), dim=-1, keepdim=True)
	# print(sim_mat.shape, sim_neg.shape, sim_mix.shape)
	# print(sim_neg.max(), sim_mix.max())


	#########################################################################################
	# los_con[i][j] = -log( exp(ziT.zj) / (exp(ziT.zj) + sum over zneg exp(ziT.zneg)) ) if j is positive for i else zero
	#########################################################################################
	los_con = -torch.log(sim_exp / (sim_exp + sim_neg)) * idx_pos
	# los_con = torch.log(1 + sim_neg/sim_exp) * idx_pos
	# los_con = (torch.log(sim_exp + sim_neg) - sim_mat) * idx_pos
	# print('los_con:', los_con.min(), los_con.max())

	#########################################################################################
	#	mean outside of log
	#########################################################################################
	# print(los_con.shape, idx_pos.shape)
	query_loss = torch.sum(los_con, dim=-1)
	query_card = torch.sum(idx_pos, dim=-1)
	# print('query_card:', query_card.min(), query_card.max())
	query_loss = query_loss / query_card.clamp_min(1)
	return torch.mean(query_loss)

def infonce_prod(q, k, temperature=0.1):
	# positive logits: Nx1
	l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

	# negative logits: NxK
	l_neg = torch.einsum('mc,nc->mn', [q, k])
	# print(l_pos.shape, l_neg.shape)

	# logits: Nx(1+K)
	# apply temperature
	logits = torch.cat([l_pos, l_neg], dim=1) / temperature

	# labels: positive key indicators, 0(N*1)表示正样本坐标为每一行的0位置
	labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
	return F.cross_entropy(logits, labels)/10

def infonce_matrix3(q, k, temperature=0.1):
	# positive logits: Nx1
	qf, qe, qb = torch.chunk(q, 2)
	kf, ke, kb = torch.chunk(k, 2)
	l_pos = torch.einsum('nc,kc->nk', [qf, kf]) + torch.einsum('nc,kc->nk', [qb, kb]) + \
			torch.einsum('nc,kc->nk', [qe, ke]) + torch.einsum('nc,kc->nk', [qe, kb]) + torch.einsum('nc,kc->nk', [qb, ke])
	l_pos = torch.exp(l_pos / temperature)

	# negative logits: NxK
	l_neg = torch.einsum('nc,kc->nk', [qf, kb]) + torch.einsum('nc,kc->nk', [qf, ke]) + \
			torch.einsum('nc,kc->nk', [qb, kf]) + torch.einsum('nc,kc->nk', [qe, kf])
	l_neg = torch.exp(l_neg / temperature).sum(dim=1, keepdim=True)
	# print(l_pos.shape, l_neg.shape)
	
	return - torch.log(l_pos / (l_pos + l_neg)).mean()

def similar_prod(q, k, temperature=0.1):
	# negative logits: NxK
	l_neg = torch.einsum('nc,nc->n', [q, k]) 
	# print(l_pos.shape, l_neg.shape)
	return 1 - l_neg.mean()

def similar_matrix2(q, k, temperature=0.1):#负太多了
	qf, qb = torch.chunk(q, 2)
	kf, kb = torch.chunk(k, 2)

	# negative logits: NxK
	l_pos = torch.einsum('nc,kc->nk', [qf, kf]) + torch.einsum('nc,kc->nk', [qb, kb])
	# print(l_pos.shape, l_neg.shape)
	return 1 - l_pos.mean()

def similar_matrix3(q, k, temperature=0.1):#负太多了
	qf, qb = torch.chunk(q, 2)
	kf, kb = torch.chunk(k, 2)

	qf, qe, qb = torch.chunk(q, 2)
	kf, ke, kb = torch.chunk(k, 2)
	l_pos = torch.einsum('nc,kc->nk', [qf, kf]) + torch.einsum('nc,kc->nk', [qb, kb]) + \
			torch.einsum('nc,kc->nk', [qe, ke]) + torch.einsum('nc,kc->nk', [qe, kb]) + torch.einsum('nc,kc->nk', [qb, ke])
	# print(l_pos.shape, l_neg.shape)
	return 1 - l_pos.mean()

CLLOSSES = {
	'nce':infonce_prod, 'sim':similar_prod,
	'nce2':infonce_matrix2, 'sim2':similar_matrix2,
	'nce3':infonce_matrix2, 'sim3':similar_matrix2,
	'':None, 'au':align_uniform
	}

class ProjectionSiam(nn.Module):
	def __init__(self, in_dim=256, inner_dim=128, out_dim=64, num_layers=2):
		super(ProjectionSiam, self).__init__()
		
		# hidden layers
		linear_hidden = [nn.Identity()]
		for i in range(num_layers - 1):
			linear_hidden.append(nn.Linear(in_dim if i == 0 else inner_dim, inner_dim))
			linear_hidden.append(nn.BatchNorm1d(inner_dim))
			linear_hidden.append(nn.ReLU(inplace=True))
		self.linear_hidden = nn.Sequential(*linear_hidden)

		self.linear_out = nn.Linear(in_dim if num_layers == 1 else inner_dim, out_dim) if num_layers >= 1 else nn.Identity()

	def forward(self, x):
		x = self.linear_hidden(x)
		x = self.linear_out(x)
		return F.normalize(x, p=2, dim=-1)

class SIAM(nn.Module):
	__name__ = 'siam'
	def __init__(self,
				 encoder,
				 clloss='nce',
				 temperature=0.1,
				 proj_num_layers=2,
				 pred_num_layers=2,
				 proj_num_length=64,
				 **kwargs):
		super().__init__()
		self.loss = CLLOSSES[clloss]
		self.encoder = encoder
		self.__name__ = 'X'.join([self.__name__, self.encoder.__name__, clloss]) 
		
		self.temperature = temperature
		self.proj_num_layers = proj_num_layers
		self.pred_num_layers = pred_num_layers

		self.predictor = ProjectionSiam(in_dim=proj_num_length, num_layers=pred_num_layers, out_dim=proj_num_length)

		# # create the queue
		# self.register_buffer("queue1", torch.randn(proj_num_length, self.num_negative))
		# self.register_buffer("queue2", torch.randn(proj_num_length, self.num_negative))
		# self.queue1 = F.normalize(self.queue1, dim=0)
		# self.queue2 = F.normalize(self.queue2, dim=0)

		# self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

	def constraint(self, **args):
		return self.encoder.constraint(**args)

	def regular(self, sampler, lab, fov=None):
		#contrastive loss split by batchsize, 07-19 I find this wrong. proj1 and proj2 not split by batchsize, but prob
		self.encoder.regular(sampler, lab, fov, return_loss=False)
		pred = self.predictor(self.encoder.emb)

		proj1, proj2 = torch.chunk(self.encoder.emb, chunks=2, dim=0)#split vessel & retinal
		pred1, pred2 = torch.chunk(pred, chunks=2, dim=0)
		# print(img1.shape, img2.shape)

		# compute loss
		losSG = self.loss(pred1, proj2.detach(), temperature=self.temperature) \
			+ self.loss(pred2, proj1.detach(), temperature=self.temperature)
		# losAU = align_uniform(pred1, proj2.detach()) + align_uniform(pred2, proj1.detach())
		return losSG# + losAU
		# return losCL + losSG

	def forward(self, img, **args):#只需保证两组数据提取出的特征标签顺序一致即可
		out = self.encoder(img, **args)
		self.pred = self.encoder.pred
		self.feat = self.encoder.feat
		# self._dequeue_and_enqueue(proj1_ng, proj2_ng)
		return out
#end#    

	# @torch.no_grad()
	# def _dequeue_and_enqueue(self, keys1, keys2):
	# 	# gather keys before updating queue
	# 	# keys1 = dist_collect(keys1)
	# 	# keys2 = dist_collect(keys2)

	# 	batch_size = keys1.shape[0]

	# 	ptr = int(self.queue_ptr)
	# 	assert self.num_negative % batch_size == 0, batch_size  # for simplicity

	# 	# replace the keys at ptr (dequeue and enqueue)
	# 	self.queue1[:, ptr:ptr + batch_size] = keys1.T
	# 	self.queue2[:, ptr:ptr + batch_size] = keys2.T
	# 	ptr = (ptr + batch_size) % self.num_negative  # move pointer

	# 	self.queue_ptr[0] = ptr

	# def self.loss(self, q, k, queue=None):
	# 	# positive logits: Nx1
	# 	l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

	# 	# negative logits: NxK
	# 	if queue is None:
	# 		l_neg = torch.einsum('nc,kc->nk', [q, k])
	# 	else:
	# 		l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])
	# 	# print(l_pos.shape, l_neg.shape)

	# 	# logits: Nx(1+K)
	# 	# apply temperature
	# 	logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature

	# 	# labels: positive key indicators, 0(N*1)表示正样本坐标为每一行的0位置
	# 	labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
	# 	return F.cross_entropy(logits, labels)



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

	
	net = SIAM(encoder=DoU())

	sampler = MLPSampler(top=4, low=0, mode='hard', temp=0.1, ver='v3')


	net.eval()
	ys = net(torch.rand_like(pred))

	for y in ys:
		print(y.shape)
	# print(net.__name__, y['loss'])

	# net.train()
	l = net.regular(sampler, pred, mask)
	print(net.__name__, l.item())


	# plot(net.emb)