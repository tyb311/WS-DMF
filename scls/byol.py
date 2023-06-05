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
# from sample import *

#start#
import torch.distributed as dist
from diffdist import functional

def dist_collect(x):
	""" collect all tensor from all GPUs
	args:
		x: shape (mini_batch, ...)
	returns:
		shape (mini_batch * num_gpu, ...)
	"""
	x = x.contiguous()
	out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype).contiguous()
				for _ in range(dist.get_world_size())]
	out_list = functional.all_gather(out_list, x)
	return torch.cat(out_list, dim=0).contiguous()

def loss_prod_infonce(q, k, temperature=0.2):
	# positive logits: Nx1
	l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

	# negative logits: NxK
	l_neg = torch.einsum('nc,kc->nk', [q, k])
	# print(l_pos.shape, l_neg.shape)

	# logits: Nx(1+K)
	# apply temperature
	logits = torch.cat([l_pos, l_neg], dim=1) / temperature

	# labels: positive key indicators, 0(N*1)表示正样本坐标为每一行的0位置
	labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
	return F.cross_entropy(logits, labels)/10

def loss_matrix_infonce(q, k, temperature=0.2):
	# positive logits: Nx1
	qf, qb = torch.chunk(q, 2)
	kf, kb = torch.chunk(k, 2)
	l_pos = torch.einsum('nc,kc->nk', [qf, kf]) + torch.einsum('nc,kc->nk', [qb, kb])
	l_pos = torch.exp(l_pos / temperature)

	# negative logits: NxK
	l_neg = torch.einsum('nc,kc->nk', [qf, kb]) + torch.einsum('nc,kc->nk', [qb, kf])
	l_neg = torch.exp(l_neg / temperature)
	# print(l_pos.shape, l_neg.shape)
	
	return - torch.log(l_pos / (l_pos + l_neg)).mean()

def loss_prod_similarity(q, k, temperature=0.2):
	# negative logits: NxK
	l_neg = torch.einsum('nc,nc->n', [q, k]) 
	# print(l_pos.shape, l_neg.shape)
	return 1 - l_neg.mean()

def loss_matrix_similarity(q, k, temperature=0.2):#负太多了
	qf, qb = torch.chunk(q, 2)
	kf, kb = torch.chunk(k, 2)

	# negative logits: NxK
	l_neg = torch.einsum('nc,kc->nk', [qf, kf]) + torch.einsum('nc,kc->nk', [qb, kb])
	# print(l_pos.shape, l_neg.shape)
	return 1 - l_neg.mean()

CLLOSSES = {
	'nce':loss_prod_infonce, 'sim':loss_prod_similarity,
	'ncem':loss_matrix_infonce, 'simm':loss_matrix_similarity,
	'':None
	}

class MLPProject(nn.Module):
	def __init__(self, in_dim=256, inner_dim=256, out_dim=128, num_layers=2):
		super(MLPProject, self).__init__()
		
		# hidden layers
		linear_hidden = [nn.Identity()]
		for i in range(num_layers - 1):
			linear_hidden.append(nn.Linear(in_dim if i == 0 else inner_dim, inner_dim))
			linear_hidden.append(nn.BatchNorm1d(inner_dim))
			linear_hidden.append(nn.LeakyReLU())
		self.linear_hidden = nn.Sequential(*linear_hidden)

		self.linear_out = nn.Linear(in_dim if num_layers == 1 else inner_dim, out_dim) if num_layers >= 1 else nn.Identity()

	def forward(self, x):
		x = self.linear_hidden(x)
		x = self.linear_out(x)

		return x

class SIAM(nn.Module):
	__name__ = 'siam'
	def __init__(self,
				 encoder,
				 clloss='nce',
				 temperature=0.2,
				 proj_num_layers=2,
				 pred_num_layers=2,
				 proj_num_length=64,
				 **kwargs):
		super().__init__()
		self.loss = CLLOSSES[clloss]
		self.encoder = encoder
		self.__name__ = '_'.join([self.__name__, self.encoder.__name__, clloss]) 
		
		self.temperature = temperature
		self.proj_num_layers = proj_num_layers
		self.pred_num_layers = pred_num_layers

		self.projector = MLPProject(in_dim=self.encoder.num_features, num_layers=proj_num_layers, out_dim=proj_num_length)
		self.projector_k = MLPProject(in_dim=self.encoder.num_features, num_layers=proj_num_layers, out_dim=proj_num_length)
		self.predictor = MLPProject(in_dim=proj_num_length, num_layers=pred_num_layers, out_dim=proj_num_length)

	sampler = None
	def regular(self, lab, fov=None):
		emb = self.sampler.select(self.feat, self.pred.detach(), lab, fov)
		emb = self.projection(emb)
		return self.sampler.infonce(emb)

	def forward(self, img, lab, **args):#只需保证两组数据提取出的特征标签顺序一致即可
		if img.shape[0]==1 or not self.training:
			output1 = self.encoder(img, lab, **args)
			out1, emb1 = output1['pred'], output1['proj']
			# print(torch.cat(emb1, dim=0).shape)
			proj1 = self.projector(emb1)
			pred1 = self.predictor(proj1)
			pred1 = F.normalize(pred1, dim=1)
			self.emb = pred1
			return {'pred':out1, 'proj':pred1, 'loss':0}

		img1, img2 = torch.chunk(img, chunks=2, dim=0)
		lab1, lab2 = torch.chunk(lab, chunks=2, dim=0)
		# print(img1.shape, img2.shape)
		
		output1 = self.encoder(img1, lab1, **args)
		out1, emb1 = output1['pred'], output1['proj']
		# print(torch.cat(emb1, dim=0).shape)
		proj1 = self.projector(emb1)
		pred1 = F.normalize(self.predictor(proj1), dim=1)

		output2 = self.encoder(img2, lab2, **args)
		out2, emb2 = output2['pred'], output2['proj']
		proj2 = self.projector(emb2)
		pred2 = F.normalize(self.predictor(proj2), dim=1)

		# print('contrastive:', pred1.shape, proj2_ng.shape)
		# compute loss
		loss = self.loss(pred1, proj2.detach(), temperature=self.temperature) \
			+ self.loss(pred2, proj1.detach(), temperature=self.temperature)

		self.emb = pred1
		return {'loss':loss, 'pred':torch.cat([out1, out2], dim=0)}
#end#    

import copy
class BYOL(nn.Module):
	__name__ = 'byol'
	def __init__(self,
				 encoder,
				 clloss='nce',
				 momentum=0.99,
				 temperature=0.2,
				 num_negative=512,
				 proj_num_layers=2,
				 pred_num_layers=2,
				 proj_num_length=64,
				 **kwargs):
		super().__init__()
		self.loss = CLLOSSES[clloss]
		self.encoder = encoder
		self.encoder_k = copy.deepcopy(encoder)
		self.__name__ = '_'.join([self.__name__, self.encoder.__name__, clloss]) 
		
		self.momentum = momentum
		self.temperature = temperature
		self.num_negative = num_negative
		
		self.proj_num_layers = proj_num_layers
		self.pred_num_layers = pred_num_layers

		self.projector = MLPProject(in_dim=self.encoder.num_features, num_layers=proj_num_layers, out_dim=proj_num_length)
		self.projector_k = MLPProject(in_dim=self.encoder.num_features, num_layers=proj_num_layers, out_dim=proj_num_length)
		self.predictor = MLPProject(in_dim=proj_num_length, num_layers=pred_num_layers, out_dim=proj_num_length)

		for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
			param_k.data.copy_(param_q.data)  # initialize
			param_k.requires_grad = False  # not update by gradient

		for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
			param_k.data.copy_(param_q.data)
			param_k.requires_grad = False

		# # create the queue
		# self.register_buffer("queue1", torch.randn(proj_num_length, self.num_negative))
		# self.register_buffer("queue2", torch.randn(proj_num_length, self.num_negative))
		# self.queue1 = F.normalize(self.queue1, dim=0)
		# self.queue2 = F.normalize(self.queue2, dim=0)

		# self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

	@torch.no_grad()
	def _momentum_update_key_encoder(self):
		_contrast_momentum = 0.99
		for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
			param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

		for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
			param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

	def forward(self, img, lab, **args):#只需保证两组数据提取出的特征标签顺序一致即可
		if img.shape[0]==1 or not self.training:
			output1 = self.encoder(img, lab, **args)
			out1, emb1 = output1['pred'], output1['proj']
			# print(torch.cat(emb1, dim=0).shape)
			proj1 = self.projector(emb1)
			pred1 = self.predictor(proj1)
			pred1 = F.normalize(pred1, dim=1)
			self.emb = pred1
			return {'pred':out1, 'proj':pred1, 'loss':0}

		img1, img2 = torch.chunk(img, chunks=2, dim=0)
		lab1, lab2 = torch.chunk(lab, chunks=2, dim=0)
		# print(img1.shape, img2.shape)
		
		output1 = self.encoder(img1, lab1, **args)
		out1, emb1 = output1['pred'], output1['proj']
		# print(torch.cat(emb1, dim=0).shape)
		proj1 = self.projector(emb1)
		pred1 = self.predictor(proj1)
		pred1 = F.normalize(pred1, dim=1)

		output2 = self.encoder(img2, lab2, **args)
		out2, emb2 = output2['pred'], output2['proj']
		proj2 = self.projector(emb2)
		pred2 = self.predictor(proj2)
		pred2 = F.normalize(pred2, dim=1)

		# compute key features
		with torch.no_grad():  # no gradient to keys
			self._momentum_update_key_encoder()  # update the key encoder

			emb1_ng = self.encoder_k(img1, lab1, **args)['proj']
			proj1_ng = self.projector_k(emb1_ng)
			proj1_ng = F.normalize(proj1_ng, dim=1)

			emb2_ng = self.encoder_k(img2, lab2, **args)['proj']
			proj2_ng = self.projector_k(emb2_ng)
			proj2_ng = F.normalize(proj2_ng, dim=1)

		# print('contrastive:', pred1.shape, proj2_ng.shape)
		# compute loss
		loss = self.loss(pred1, proj2_ng.detach(), temperature=self.temperature) \
			+ self.loss(pred2, proj1_ng.detach(), temperature=self.temperature)
		# loss = self.self.loss(pred1, proj2_ng, self.queue2) \
		#     + self.self.loss(pred2, proj1_ng, self.queue1)

		# self._dequeue_and_enqueue(proj1_ng, proj2_ng)
		self.emb = pred1
		# self.emb = [torch.cat([e,f], dim=0) for e,f in zip(torch.chunk(pred1, 4), torch.chunk(pred2, 4))]
		# self.emb = [F.normalize(self.projector(e), dim=1) for e in emb]
		return {'loss':loss, 'pred':torch.cat([out1, out2], dim=0)}

'''

	@torch.no_grad()
	def _dequeue_and_enqueue(self, keys1, keys2):
		# gather keys before updating queue
		# keys1 = dist_collect(keys1)
		# keys2 = dist_collect(keys2)

		batch_size = keys1.shape[0]

		ptr = int(self.queue_ptr)
		assert self.num_negative % batch_size == 0, batch_size  # for simplicity

		# replace the keys at ptr (dequeue and enqueue)
		self.queue1[:, ptr:ptr + batch_size] = keys1.T
		self.queue2[:, ptr:ptr + batch_size] = keys2.T
		ptr = (ptr + batch_size) % self.num_negative  # move pointer

		self.queue_ptr[0] = ptr

	def self.loss(self, q, k, queue=None):
		# positive logits: Nx1
		l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

		# negative logits: NxK
		if queue is None:
			l_neg = torch.einsum('nc,kc->nk', [q, k])
		else:
			l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])
		# print(l_pos.shape, l_neg.shape)

		# logits: Nx(1+K)
		# apply temperature
		logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature

		# labels: positive key indicators, 0(N*1)表示正样本坐标为每一行的0位置
		labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
		return F.cross_entropy(logits, labels)
		'''