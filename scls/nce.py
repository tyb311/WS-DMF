#start#
import os, glob, sys, time, torch
import numpy as np
from torch import nn
from torch.nn import functional as F

#start#
def info_nce(true, feat, temperature=0.1):
	"""
	“Contrastive Semi-Supervised Learning for 2D Medical Image Segmentation,” arXiv:2106.06801 [cs]
	true is one hot class vector of patches (S, num_classes)
	feat = (S, dim)
	"""
	# idx_pos[i][j] = (class[i] == class[j])
	idx_pos = torch.mm(true, true.permute(1,0))

	feat = F.normalize(feat, p=2, dim=-1)
	# sim_mat[i][j] = cosine angle between ith and jth embed
	sim_mat = torch.mm(feat, feat.clone().detach().permute(1,0)) / temperature
	
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

def info_nce2(true, feat, temperature=0.1):
	los1 = info_nce(true, feat, temperature=temperature)
	los2 = info_nce(1-true, feat, temperature=temperature)
	# print(los1.item(), los2.item())
	return (los1 + los2)*0.5

def align_uniform(true, feat):
	# “Understanding contrastive representation learning through alignment and uniformity on the hypersphere,” ICML 2020
	#https://github.com/SsnL/align_uniform
	def align_loss(x, y, alpha=2):
		assert x.shape==y.shape, "shape of postive sample must be the same"
		return (x - y).norm(p=2, dim=1).pow(alpha).mean()

	def uniform_loss(x, t=2):
		return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
		#pdist If input has shape N×M then the output will have shape 1/2 * N * (N - 1) 

	# feat = F.normalize(feat, p=2, dim=-1)
	# fgd,fl,bh,bl = torch.chunk(feat, chunks=4, dim=0)
	# fgd = feat[true.repeat(1, feat.shape[1])>0.5]
	idx_pos = (true>0.5).type(torch.bool)
	fgd = torch.masked_select(feat, mask=idx_pos).reshape(-1, feat.shape[-1])
	bgd = torch.masked_select(feat, mask=~idx_pos).reshape(-1, feat.shape[-1])
	loss_align = align_loss(fgd, torch.flip(fgd, dims=[0])) + align_loss(bgd, torch.flip(bgd, dims=[0]))
	loss_unif = uniform_loss(fgd) + uniform_loss(bgd)
	# loss_unif = uniform_loss(feat)
	# print(loss_align.item(), loss_unif.item())
	return loss_align + loss_unif

def ce_nce(true, feat, temperature=0.1):
	idx_pos = (true>0.5).type(torch.bool)
	fgd = torch.masked_select(feat, mask=idx_pos).reshape(-1, feat.shape[-1])
	bgd = torch.masked_select(feat, mask=~idx_pos).reshape(-1, feat.shape[-1])

	# positive logits: Nx1
	# l_pos = fgd @ fgd.permute(1,0).mean(dim=1).unsqueeze(-1)
	l_pos = (fgd*torch.flip(fgd, dims=[0]).detach()).sum(dim=1, keepdim=True)
	# negative logits: NxK
	l_neg = fgd @ bgd.permute(1,0)
	# print(l_pos.shape, l_neg.shape)

	# logits: Nx(1+K)
	logits = torch.cat([l_pos, l_neg], dim=1) /temperature

	# labels: positive key indicators
	labels = torch.zeros(logits.shape[0], dtype=torch.long).to(feat.device)

	return F.cross_entropy(logits, labels)

def ce_nce2(true, feat, temperature=0.1):
	los1 = ce_nce(true, feat, temperature=temperature)
	los2 = ce_nce(1-true, feat, temperature=temperature)
	# print(los1.item(), los2.item())
	return (los1 + los2)*0.5

class ConLoss(nn.Module):
	def __init__(self, con='nce', temp=0.1):
		super(ConLoss, self).__init__()
		self.temp = temp
		if con=='nce':
			self.func = info_nce
		elif con=='nce2':
			self.func = info_nce2
		elif con=='au':
			self.func = align_uniform
		elif con=='ce':
			self.func = ce_nce
		elif con=='ce2':
			self.func = ce_nce2
		else:
			raise NotImplementedError

	def forward(self, feat, *args):
		feat = F.normalize(feat, p=2, dim=-1)
		true = torch.cat([
			torch.ones(size=(feat.shape[0]//2, 1), dtype=torch.float32),
			torch.zeros(size=(feat.shape[0]//2, 1), dtype=torch.float32)
		], dim=0).to(feat.device)
		return self.func(true, feat)
#end#

if __name__ == '__main__':
	feat = torch.rand(256, 64)
	true = torch.rand(256, 1).round()
	# print(info_nce(true, feat).item())
	# print(info_nce2(true, feat).item())
	# print(align_uniform(true, feat).item())
	
	crit = ConLoss(con='au')
	print(crit(feat).item())

	crit = ConLoss(con='nce')
	print(crit(feat).item())
	crit = ConLoss(con='nce2')
	print(crit(feat).item())

	crit = ConLoss(con='ce')
	print(crit(feat).item())
	crit = ConLoss(con='ce2')
	print(crit(feat).item())
# info_nce然我看到了真正的info_nce，但是它没有考虑背景像素
# 或者可以对血管像素做info_nce，对背景像素做align_uniform


	# # normalize就是除以F范数
	# x1 = F.normalize(feat, p=2, dim=1)
	# x2 = feat / feat.norm(p=2, dim=1, keepdim=True)
	# print((x1-x2).abs().max())
	# print((x1==x2).min())

	# x1 = feat @ feat.permute(1,0)
	# x2 = torch.mm(feat, feat.permute(1,0))
	# x3 = torch.einsum('nc,mc->nm', [feat, feat])
	# print((x1==x2).min())
	# print((x1==x3).min())
	# print((x3==x2).min())
