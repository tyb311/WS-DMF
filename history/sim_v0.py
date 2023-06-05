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

def compute_kl_loss(p, q):
	p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='sum')
	q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='sum')
	return (p_loss + q_loss) / 2

def similar_prod(q, k, temperature=0.1):
	# negative logits: NxK
	l_neg = torch.einsum('nc,nc->n', [q, k]) 
	# print(l_pos.shape, l_neg.shape)
	return 1 - l_neg.mean()

#start#
def similar_matrix2(q, k, temperature=0.1):#负太多了
	# print('similar_matrix2:', q.shape, k.shape)
	qf, qb = torch.chunk(q, 2, dim=0)
	kf, kb = torch.chunk(k, 2, dim=0)

	# negative logits: NxK
	l_pos = torch.einsum('nc,kc->nk', [qf, kf])
	l_neg = torch.einsum('nc,kc->nk', [qb, kb])
	# print(l_pos.shape, l_neg.shape)
	return 2 - l_pos.mean() - l_neg.mean()

def similar_matrix3(q, k, temperature=0.1):#负太多了
	return 1 - q @ k.permute(1,0).mean()

def align_loss(x, y, alpha=2):
	assert x.shape==y.shape, "shape of postive sample must be the same"
	return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
	return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
	#pdist If input has shape N×M then the output will have shape 1/2 * N * (N - 1) 

def align_uniform(q, k, temperature=0.1):
	# “Understanding contrastive representation learning through alignment and uniformity on the hypersphere,” ICML 2020
	#https://github.com/SsnL/align_uniform
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
	# los_con = -torch.log(sim_exp / (sim_exp + sim_neg)) * idx_pos
	# los_con = torch.log(1 + sim_neg/sim_exp) * idx_pos
	los_con = (torch.log(sim_exp + sim_neg) - sim_mat) * idx_pos
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
	qf, qe, qb = torch.chunk(q, 3)
	kf, ke, kb = torch.chunk(k, 3)
	los1 = infonce_matrix2(torch.cat([qf, qe], dim=0), torch.cat([kf, ke], dim=0))
	los2 = infonce_matrix2(torch.cat([qf, qb], dim=0), torch.cat([kf, kb], dim=0))
	return los1 + los2
	# l_pos = torch.einsum('nc,kc->nk', [qf, kf]) + torch.einsum('nc,kc->nk', [qb, kb]) + \
	# 		torch.einsum('nc,kc->nk', [qe, ke]) + torch.einsum('nc,kc->nk', [qe, kb]) + torch.einsum('nc,kc->nk', [qb, ke])
	# l_pos = torch.exp(l_pos / temperature)

	# # negative logits: NxK
	# l_neg = torch.einsum('nc,kc->nk', [qf, kb]) + torch.einsum('nc,kc->nk', [qf, ke]) + \
	# 		torch.einsum('nc,kc->nk', [qb, kf]) + torch.einsum('nc,kc->nk', [qe, kf])
	# l_neg = torch.exp(l_neg / temperature).sum(dim=1, keepdim=True)
	# # print(l_pos.shape, l_neg.shape)
	
	# return - torch.log(l_pos / (l_pos + l_neg)).mean()

CLLOSSES = {
	# 'nce2':infonce_matrix2, 'nce3':infonce_matrix2, 
	'sim2':similar_matrix2, 'sim3':similar_matrix2,
	'':None, 'au':align_uniform
	# 'nce':infonce_prod, 'sim':similar_prod, 
	}
#end#  

	# def render(self, feat, prob, margin=0.15):
	# 	B, C, H, W = feat.shape
	# 	with torch.no_grad():#渲染时候投影模块去除梯度，但渲染模块需要学习
	# 		mask = ((prob.detach()-0.5).abs()<margin).type(torch.bool)
	# 		feat = feat.permute(0,2,3,1).reshape(-1,C)[mask.permute(0,2,3,1).reshape(-1,1).repeat(1,C)]
	# 		# print('feat:', feat.shape)
	# 		proj = self.forward(feat.reshape(-1,C))
	# 		# proj = self.tsne.fit_transform(rend)
	# 		# print(prob.shape, mask.shape, proj.shape, proj.numel())
	# 		rend = torch.zeros(B, self.dim_rend, H, W).to(feat.device)
	# 		rend[mask.repeat(1,self.dim_rend,1,1)] = self.mlp_render(proj).view(-1)
	# 	return rend


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

	
	net = SIAM(encoder=lunet(), clloss='sim2')

	sampler = MLPSampler(top=4, low=0, mode='prob')


	# net.eval()
	ys = net(torch.rand_like(pred))

	for y in ys:
		print(y.shape)
	# print(net.__name__, y['loss'])

	# net.train()
	l = net.regular(sampler, pred, mask)
	# l = net.regular3(sampler, pred, mask)
	print(net.__name__, l.item())


	# plot(net.emb)