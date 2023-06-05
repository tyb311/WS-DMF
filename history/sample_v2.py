import torch
import torch.nn as nn
import torch.nn.functional as F


#start#
def points_selection_hard(feat, prob, mask, k=200, **args):#point selection by ranking
	assert len(feat.shape)==2, 'feat should contains N*L two dims!'
	L = feat.shape[-1]
	# print(feat.shape, mask.shape)
	feat = feat[mask.view(-1,1).repeat(1, L)>.5].view(-1, L)
	with torch.no_grad():
		prob = prob[mask>.5].view(-1)
		idx = torch.sort(prob, dim=-1, descending=True)[1]
	h = torch.index_select(feat, dim=0, index=idx[:k])
	l = torch.index_select(feat, dim=0, index=idx[-k:])
	# print('lh:', l.shape, h.shape)
	# print(prob[idx[:k]].view(-1)[:9])
	# print(prob[idx[-k:]].view(-1)[:9])
	return h, l

def points_selection_half(feat, prob, mask, k=200, **args):#point selection by ranking
	assert len(feat.shape)==2, 'feat should contains N*L two dims!'
	L = feat.shape[-1]
	# print(feat.shape, mask.shape)
	feat = feat[mask.view(-1,1).repeat(1, L)>.5].view(-1, L)
	with torch.no_grad():
		prob = prob[mask>.5].view(-1)
		idx = torch.sort(prob, dim=-1, descending=False)[1]
		idx_l, idx_h = torch.chunk(idx, chunks=2, dim=0)

		sample1 = idx_h[torch.randperm(idx_h.shape[0])[:k]]
		sample2 = idx_l[torch.randperm(idx_l.shape[0])[:k]]
	# print(prob[sample][:9])
	h = torch.index_select(feat, dim=0, index=sample1)
	# print(prob[sample][:9])
	l = torch.index_select(feat, dim=0, index=sample2)
	# print('lh:', l.shape, h.shape)
	return h, l

def points_selection_rand(feat, prob, mask, k=200, **args):#point selection by ranking
	assert len(feat.shape)==2, 'feat should contains N*L two dims!'
	L = feat.shape[-1]
	# print(feat.shape, mask.shape)
	feat = feat[mask.view(-1,1).repeat(1, L)>.5].view(-1, L)
	with torch.no_grad():
		prob = prob[mask>.5].view(-1)
		idx = torch.sort(prob, dim=-1, descending=False)[1]
		rand = torch.randperm(idx.shape[0])
		sample1 = idx[rand[:k]]
		# print(prob[sample][:9])
		sample2 = idx[rand[-k:]]
		# print(prob[sample][:9])
	h = torch.index_select(feat, dim=0, index=sample1)
	l = torch.index_select(feat, dim=0, index=sample2)
	# print('lh:', l.shape, h.shape)
	return h, l

def points_selection_part(feat, prob, mask, k=200, top=4, low=2):#point selection by ranking
	assert len(feat.shape)==2, 'feat should contains N*L two dims!'
	assert top>=0 and top<5, 'top must be in range(0,5)'
	assert low>=0 and low<5, 'low must be in range(0,5)'
	L = feat.shape[-1]
	# print(feat.shape, mask.shape)
	feat = feat[mask.view(-1,1).repeat(1, L)>.5].view(-1, L)
	with torch.no_grad():
		prob = prob[mask>.5].view(-1)
		idx = torch.sort(prob, dim=-1, descending=False)[1]
		hard_ranks = torch.chunk(idx, chunks=5, dim=0)
		sample1 = hard_ranks[top][torch.randperm(hard_ranks[top].shape[0])[:k]]
		# print(prob[sample][:9])
		sample2 = hard_ranks[low][torch.randperm(hard_ranks[low].shape[0])[:k]]
		# print(prob[sample][:9])
	h = torch.index_select(feat, dim=0, index=sample1)
	l = torch.index_select(feat, dim=0, index=sample2)
	# print('lh:', l.shape, h.shape)
	return h, l
	
class MLPSampler:
	func = points_selection_part
	def __init__(self, top=4, low=1, mode='hard', temp=0.2, ver='v2'):
		self.top = top
		self.low = low
		self.temp = temp
		if mode=='part':
			self.func = points_selection_part
		elif mode=='rand':
			self.func = points_selection_rand
		elif mode=='half':
			self.func = points_selection_half
		elif mode=='hard':
			self.func = points_selection_hard

		if ver=='v1':
			self.infonce = self.infonce_v1
		elif ver=='v2':
			self.infonce = self.infonce_v2
		# elif ver=='v3':
		# 	self.infonce = self.infonce_v3
		elif ver=='au':
			self.infonce = self.align_uniform
		elif ver=='cc':
			self.infonce = self.clcr
		else:
			self.infonce = self.infonce_v0

	def softmax(self, pos, neg):
		pos = torch.exp(pos/self.temp)
		neg = torch.exp(neg/self.temp)
		if len(pos.shape)==2:
			neg = neg.sum(dim=-1)
		# print(pos.shape, neg.shape)
		los = -torch.log(pos / (pos + neg + 1e-5)).mean()
		return los
	
	# def infonce_v0(self, emb):
	# 	#V0:info-nce
	# 	f, b = torch.chunk(emb, 2, dim=0)
	# 	neg = (f * b).sum(-1)
	# 	pos = (f * f.clone().detach()).sum(-1) + (b * b.clone().detach()).sum(-1)
	# 	return self.softmax(pos, neg)
	
	def infonce_v1(self, emb):
		#V0:info-nce
		f, b = torch.chunk(emb, 2, dim=0)
		neg = f @ b.permute(1,0)
		pos = f.clone().detach() @ f.permute(1,0) + b.clone().detach() @ b.permute(1,0)
		return self.softmax(pos, neg)

	def infonce_v2(self, emb):
		#V0:info-nce
		f, b = torch.chunk(emb, 2, dim=0)
		neg = f @ b.permute(1,0)
		pos1 = f.clone().detach() @ f.permute(1,0)
		pos2 = b.clone().detach() @ b.permute(1,0)
		los1, los2 = self.softmax(pos1, neg), self.softmax(pos2, neg)
		# print(los1, los2)
		return (los1 + los2)/6

	# def infonce_v3(self, emb):
	# 	#V0:info-nce
	# 	f, e, b = torch.chunk(emb, 3, dim=0)
	# 	neg = f @ b.permute(1,0) + f @ e.permute(1,0)
	# 	pos1 = f @ f.permute(1,0)
	# 	pos2 = b @ b.permute(1,0) + e @ e.permute(1,0) + e @ b.permute(1,0)
	# 	return self.softmax(pos1, neg) + self.softmax(pos2, neg)

	def clcr(self, feat):
		"""
		true is one hot class vector of patches (S, num_classes)
		feat = (S, dim)
		"""
		true = torch.zeros(size=(feat.shape[0], 1), dtype=torch.float32).to(feat.device)
		true[:feat.shape[0]//2] = 1
		# cls_eql[i][j] = (class[i] == class[j])
		cls_eql = torch.mm(true, true.permute(1,0))

		feat = F.normalize(feat, p=2, dim=-1)
		# sim_mat[i][j] = cosine angle between ith and jth embed
		sim_mat = torch.matmul(feat, feat.permute(1,0))
		sim_exp = torch.exp(sim_mat / self.temp)
		# sim_negs = sum or exp(similarity) of ith patch with every negative patch
		sim_neg = sim_exp * (1 - cls_eql)

		#########################################################################################
		# The hard negative sample z = a*zi + (1 - a)*zn. sim(z, zi) = a + (1 - a)*sim(zn, zi)
		#########################################################################################
		# alpha = torch.rand(1, dtype=torch.float32)*0.4
		# hard_sim_neg = torch.exp(alpha + (1. - alpha) * sim_mat) * (1 - cls_eql)
		sim_neg = torch.sum(sim_neg, dim=-1, keepdim=True)
		# print(sim_mat.shape, sim_neg.shape, hard_sim_neg.shape)


		#########################################################################################
		# contrast[i][j] = -log( exp(ziT.zj) / (exp(ziT.zj) + sum over zneg exp(ziT.zneg)) ) if j is positive for i else zero
		#########################################################################################
		numerator = sim_exp
		denominator = numerator + sim_neg# + hard_sim_neg

		# print('numerator:', numerator.min(), numerator.max())
		# print('denominator:', denominator.min(), denominator.max())
		contrast = -torch.log(numerator / denominator) * cls_eql
		# print('contrast:', contrast.min(), contrast.max())

		# print(numerator.shape, denominator.shape)
		# print(contrast.shape, cls_eql.shape)
		query_loss = torch.reshape(torch.sum(contrast, dim=-1), [-1, 1])
		query_card = torch.reshape(torch.sum(cls_eql, dim=-1), [-1, 1])
		# print('query_card:', query_card.min(), query_card.max())
		query_loss = query_loss / query_card.clamp_min(1e-5)
		cl_loss = torch.mean(query_loss)
		return cl_loss

	def align_uniform(self, feats):
		#https://github.com/SsnL/align_uniform
		def align_loss(x, y, alpha=2):
			return (x - y).norm(p=2, dim=1).pow(alpha).mean()

		def uniform_loss(x, t=2):
			return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
			#pdist If input has shape N×M then the output will have shape 1/2 * N * (N - 1) 
		# feats = F.normalize(feats, p=2, dim=-1)
		fh,fl,bh,bl = torch.chunk(feats, chunks=4, dim=0)
		assert fh.shape==bh.shape, "shape of high confidence must match to sum out of log"
		
		loss_align = align_loss(fh, fl) + align_loss(bh, bl)
		loss_unif = (uniform_loss(fh) + uniform_loss(fl) + uniform_loss(bh) + uniform_loss(bl)) / 4
		# print(loss_align.item(), loss_unif.item())
		return (loss_align + loss_unif)*100

	@staticmethod
	def rand(*args):
		return MLPSampler(mode='rand').select(*args)
	@staticmethod
	def half(*args):
		return MLPSampler(mode='half').select(*args)
	@staticmethod
	def hard(*args):
		return MLPSampler(mode='hard').select(*args)

	def select(self, feat, pred, mask, fov=None, ksize=5):
		# assert mode in ['hard', 'semi', 'hazy', 'edge'], 'sample selection mode wrong!'
		assert feat.shape[-2:]==mask.shape[-2:], 'shape of feat & mask donot match!'
		assert feat.shape[-2:]==pred.shape[-2:], 'shape of feat & pred donot match!'
		# reshape embeddings
		feat = feat.permute(0,2,3,1).reshape(-1, feat.shape[1])
		# feat = F.normalize(feat, p=2, dim=-1)
		mask = mask.round()
		# back = (F.max_pool2d(mask, (ksize, ksize), 1, ksize//2) - mask).round()
		back = (1-mask).round()*fov

		fh, fl = self.func(feat,   pred, mask, top=self.top, low=self.low)
		bh, bl = self.func(feat, 1-pred, back, top=self.top, low=self.low)
		# print('mlp_sample:', fh.shape, fl.shape, bh.shape, bl.shape)
		# return [fh, fl, bh, bl]
		# print(mode)
		return torch.cat([fh, fl, bh, bl], dim=0)

	def select3(self, feat, pred, mask, fov=None, ksize=5):
		# assert mode in ['hard', 'semi', 'hazy', 'edge'], 'sample selection mode wrong!'
		assert feat.shape[-2:]==mask.shape[-2:], 'shape of feat & mask donot match!'
		assert feat.shape[-2:]==pred.shape[-2:], 'shape of feat & pred donot match!'
		# reshape embeddings
		feat = feat.permute(0,2,3,1).reshape(-1, feat.shape[1])
		# feat = F.interpolate(feat, size=mask.shape[-2:], mode='bilinear', align_corners=True)
		# feat = F.normalize(feat, p=2, dim=-1)
		mask = mask.round()
		dilate = F.max_pool2d(mask, (ksize, ksize), stride=1, padding=ksize//2).round()
		edge = (dilate - mask).round()
		back = (1-dilate).round()*fov

		# plt.subplot(131),plt.imshow(mask.squeeze().data.numpy())
		# plt.subplot(132),plt.imshow(edge.squeeze().data.numpy())
		# plt.subplot(133),plt.imshow(back.squeeze().data.numpy())
		# plt.show()

		# assert back.sum()>0, 'back has no pixels!'
		# assert mask.sum()>0, 'mask has no pixels!'
		# print('mask:', mask.sum().item(), mask.sum().item()/mask.numel())
		# print('edge:', edge.sum().item(), edge.sum().item()/edge.numel())
		# print('back:', back.sum().item(), back.sum().item()/back.numel())
		# print(feat.shape, pred.shape, mask.shape)

		fh, fl = self.func(feat,   pred, mask, top=self.top, low=self.low)
		eh, el = self.func(feat, 1-pred, edge, top=self.top, low=self.low)
		bh, bl = self.func(feat, 1-pred, back, top=self.top, low=self.low)
		# print('mlp_sample:', fh.shape, fl.shape, bh.shape, bl.shape)
		# return [fh, fl, bh, bl]
		# print(mode)
		return torch.cat([fh, fl, eh, el, bh, bl], dim=0)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
def plot4(emb=None, path_save='emb.png'):
	# assert len(emb[0].shape)==2, 'no embedding!'
	# for f in emb:
	# 	print('EMB:', f.shape)
	print('save embedding!')
	tsne = TSNE(n_components=2, init='pca', random_state=2021)
	emb = tsne.fit_transform(emb.detach().cpu().data.numpy() )
	[fh, fl, bh, bl] = np.split(emb, 4, axis=0)

	# print(fh.shape, fl.shape, bh.shape, bl.shape)
	fig = plt.figure()
	plt.scatter(fh[:, 0], fh[:, 1], c='red', marker='^', linewidths=.1)
	plt.scatter(fl[:, 0], fl[:, 1], c='pink', marker='^', linewidths=.1)
	plt.scatter(bh[:, 0], bh[:, 1], c='green', marker='o', linewidths=.1)
	plt.scatter(bl[:, 0], bl[:, 1], c='lightgreen', marker='o', linewidths=.1, alpha=0.7)
	plt.title('embedded feature distribution')
	plt.legend(['vessel_easy', 'vessel_hard', 'retinal_easy', 'retinal_hard'])
	# plt.legend(['vessel_h', 'vessel_l', 'retinal_h', 'retinal_l'])
	plt.tight_layout()
	# plt.xticks([])
	# plt.yticks([])
	# plt.savefig('./draw_examples3.pdf', format='pdf')
	plt.savefig(path_save, dpi=90)
	# plt.show()
	return fig

def plot2(emb=None, path_save='emb.png'):
	# assert len(emb[0].shape)==2, 'no embedding!'
	# for f in emb:
	# 	print('EMB:', f.shape)
	print('save embedding!')
	tsne = TSNE(n_components=2, init='pca', random_state=2021)
	emb = tsne.fit_transform(emb.detach().cpu().data.numpy() )
	[f, b] = np.split(emb, 2, axis=0)

	# print(fh.shape, fl.shape, bh.shape, bl.shape)
	fig = plt.figure()
	plt.scatter(f[:, 0], f[:, 1], c='red', marker='^', linewidths=.1)
	plt.scatter(b[:, 0], b[:, 1], c='green', marker='o', linewidths=.1)
	plt.title('embedded feature distribution')
	plt.legend(['vessel', 'retinal'])
	# plt.legend(['vessel_h', 'vessel_l', 'retinal_h', 'retinal_l'])
	plt.tight_layout()
	# plt.xticks([])
	# plt.yticks([])
	# plt.savefig('./draw_examples3.pdf', format='pdf')
	plt.savefig(path_save, dpi=90)
	# plt.show()
	return fig
#end#

def plot6(emb=None, path_save='emb.png'):
	# assert len(emb[0].shape)==2, 'no embedding!'
	# for f in emb:
	# 	print('EMB:', f.shape)
	print('save embedding!')
	tsne = TSNE(n_components=2, init='pca', random_state=2021)
	emb = tsne.fit_transform(emb.detach().cpu().data.numpy() )
	[fh, fl, eh, el, bh, bl] = np.split(emb, 6, axis=0)

	# print(fh.shape, fl.shape, bh.shape, bl.shape)
	fig = plt.figure()
	plt.scatter(fh[:, 0], fh[:, 1], c='red', marker='^', linewidths=.1)
	plt.scatter(fl[:, 0], fl[:, 1], c='pink', marker='^', linewidths=.1)
	plt.scatter(eh[:, 0], eh[:, 1], c='purple', marker='o', linewidths=.1)
	plt.scatter(el[:, 0], el[:, 1], c='violet', marker='o', linewidths=.1)
	plt.scatter(bh[:, 0], bh[:, 1], c='green', marker='o', linewidths=.1)
	plt.scatter(bl[:, 0], bl[:, 1], c='lightgreen', marker='o', linewidths=.1, alpha=0.7)
	plt.title('embedded feature distribution')
	plt.legend(['vessel_easy', 'vessel_hard', 'edge_easy', 'edge_hard', 'retinal_easy', 'retinal_hard'])
	# plt.legend(['vessel_h', 'vessel_l', 'retinal_h', 'retinal_l'])
	plt.tight_layout()
	# plt.xticks([])
	# plt.yticks([])
	# plt.savefig('./draw_examples3.pdf', format='pdf')
	plt.savefig(path_save, dpi=90)
	# plt.show()
	return fig

def plot3(emb=None, path_save='emb.png'):
	# assert len(emb[0].shape)==2, 'no embedding!'
	# for f in emb:
	# 	print('EMB:', f.shape)
	print('save embedding!')
	tsne = TSNE(n_components=2, init='pca', random_state=2021)
	emb = tsne.fit_transform(emb.detach().cpu().data.numpy() )
	[f, e, b] = np.split(emb, 3, axis=0)

	# print(fh.shape, fl.shape, bh.shape, bl.shape)
	fig = plt.figure()
	plt.scatter(f[:, 0], f[:, 1], c='red', marker='^', linewidths=.1)
	plt.scatter(e[:, 0], e[:, 1], c='purple', marker='o', linewidths=.1)
	plt.scatter(b[:, 0], b[:, 1], c='green', marker='o', linewidths=.1)
	plt.title('embedded feature distribution')
	plt.legend(['vessel', 'edge', 'retinal'])
	plt.tight_layout()
	# plt.xticks([])
	# plt.yticks([])
	# plt.savefig('./draw_examples3.pdf', format='pdf')
	plt.savefig(path_save, dpi=90)
	# plt.show()
	return fig



import cv2
if __name__ == '__main__':
	pred = cv2.imread('figures/z_pred.png', cv2.IMREAD_GRAYSCALE)
	mask = cv2.imread('figures/z_mask.png', cv2.IMREAD_GRAYSCALE)
	true = cv2.imread('figures/z_true.png', cv2.IMREAD_GRAYSCALE)
	h, w = pred.shape

	pred = torch.from_numpy(pred.astype(np.float32)/255).unsqueeze(0).unsqueeze(0)
	mask = torch.from_numpy(mask.astype(np.float32)/255).unsqueeze(0).unsqueeze(0)
	true = torch.from_numpy(true.astype(np.float32)/255).unsqueeze(0).unsqueeze(0)
	print('imread:', pred.shape, mask.shape, true.shape)

	feat = torch.rand(1,32,h,w)
	feat = F.normalize(feat, p=2, dim=1)

	sample1 = MLPSampler(top=1, low=4, ver='v1')
	emb = sample1.select(feat, pred, true, mask)
	print(emb.shape)
	# print(sample1.infonce_v0(emb))
	print(sample1.infonce_v1(emb))
	print(sample1.infonce_v2(emb))
	print(sample1.align_uniform(emb))
	print(sample1.clcr(emb))
	# plot6(emb)
	# plt.show()

	emb = MLPSampler.hard(feat, pred, true, mask)
	print(emb.shape)

	# plot3(emb)
	# plt.show()

	# sample2 = MLPSampler(top=2, low=4, mode='part')
	# emb = sample2.select(feat, pred, true)
	# print(emb.shape)

	# sample3 = MLPSampler(top=2, low=4, mode='rand')
	# emb = sample3.select(feat, pred, true)
	# print(emb.shape)

	emb = MLPSampler.rand(feat, pred, true, mask)
	print(emb.shape)

	emb = MLPSampler.half(feat, pred, true, mask)
	print(emb.shape)

	# plot(emb)
	# plt.show()