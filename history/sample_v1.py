import torch
import torch.nn as nn
import torch.nn.functional as F


#start#
def points_selection_hard(feat, prob, mask, k=320, **args):#point selection by ranking
	assert len(feat.shape)==2, 'feat should contains N*L two dims!'
	L = feat.shape[-1]
	# print(feat.shape, mask.shape)
	feat = feat[mask.view(-1,1).repeat(1, L)>.5].view(-1, L)
	with torch.no_grad():
		prob = prob[mask>.5].view(-1)
		# seq, idx = torch.sort(prob, dim=-1, descending=True)
		# print(seq.shape, idx.shape, seq[0].item(), idx[0].item(), seq[-1].item(), idx[-1].item())
		idx = torch.sort(prob, dim=-1, descending=True)[1]
		# k = min(idx.numel()//3, k)
		h = torch.index_select(feat, dim=0, index=idx[:k])
		l = torch.index_select(feat, dim=0, index=idx[-k:])
		# print('lh:', l.shape, h.shape)
		# print(prob[idx[:k]].view(-1)[:9])
		# print(prob[idx[-k:]].view(-1)[:9])
	return h, l

def points_selection_half(feat, prob, mask, k=320, **args):#point selection by ranking
	assert len(feat.shape)==2, 'feat should contains N*L two dims!'
	L = feat.shape[-1]
	# print(feat.shape, mask.shape)
	feat = feat[mask.view(-1,1).repeat(1, L)>.5].view(-1, L)
	with torch.no_grad():
		prob = prob[mask>.5].view(-1)
		# print(seq.shape, idx.shape, seq[0].item(), idx[0].item(), seq[-1].item(), idx[-1].item())
		idx = torch.sort(prob, dim=-1, descending=False)[1]
		idx_l, idx_h = torch.chunk(idx, chunks=2, dim=0)

		sample = idx_h[torch.randperm(idx_h.shape[0])[:k]]
		# print(prob[sample][:9])
		h = torch.index_select(feat, dim=0, index=sample)

		sample = idx_l[torch.randperm(idx_l.shape[0])[:k]]
		# print(prob[sample][:9])
		l = torch.index_select(feat, dim=0, index=sample)
		# print('lh:', l.shape, h.shape)
	return h, l

def points_selection_rand(feat, prob, mask, k=320, **args):#point selection by ranking
	assert len(feat.shape)==2, 'feat should contains N*L two dims!'
	L = feat.shape[-1]
	# print(feat.shape, mask.shape)
	feat = feat[mask.view(-1,1).repeat(1, L)>.5].view(-1, L)
	with torch.no_grad():
		prob = prob[mask>.5].view(-1)
		# print(seq.shape, idx.shape, seq[0].item(), idx[0].item(), seq[-1].item(), idx[-1].item())
		idx = torch.sort(prob, dim=-1, descending=False)[1]
		rand = torch.randperm(idx.shape[0])
		sample = idx[rand[:k]]
		# print(prob[sample][:9])
		h = torch.index_select(feat, dim=0, index=sample)
		sample = idx[rand[-k:]]
		# print(prob[sample][:9])
		l = torch.index_select(feat, dim=0, index=sample)
		# print('lh:', l.shape, h.shape)
	return h, l

def points_selection_part(feat, prob, mask, k=320, top=4, low=2):#point selection by ranking
	assert len(feat.shape)==2, 'feat should contains N*L two dims!'
	assert top>=0 and top<5, 'top must be in range(0,5)'
	assert low>=0 and low<5, 'low must be in range(0,5)'
	L = feat.shape[-1]
	# print(feat.shape, mask.shape)
	feat = feat[mask.view(-1,1).repeat(1, L)>.5].view(-1, L)
	with torch.no_grad():
		prob = prob[mask>.5].view(-1)
		# print(seq.shape, idx.shape, seq[0].item(), idx[0].item(), seq[-1].item(), idx[-1].item())
		idx = torch.sort(prob, dim=-1, descending=False)[1]

		# very_hard, semi_hard, hazy_edge, semi_easy, very_easy = torch.chunk(idx, chunks=5, dim=0)
		hard_ranks = torch.chunk(idx, chunks=5, dim=0)

		sample = hard_ranks[top][torch.randperm(hard_ranks[top].shape[0])[:k]]
		# print(prob[sample][:9])
		h = torch.index_select(feat, dim=0, index=sample)
		sample = hard_ranks[low][torch.randperm(hard_ranks[low].shape[0])[:k]]
		# print(prob[sample][:9])
		l = torch.index_select(feat, dim=0, index=sample)
		# print('lh:', l.shape, h.shape)
	return h, l
	
class MLPSampler:
	func = points_selection_part
	def __init__(self, top=4, low=1, mode='part', temp=0.2, ver='v1'):
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

	def softmax(self, pos, neg):
		pos = torch.exp(pos/self.temp)
		neg = torch.exp(neg/self.temp)
		los = -torch.log(pos / (pos + neg + 1e-5)).mean()
		return los
	
	# @staticmethod
	def infonce_v1(self, emb):
		#V0:info-nce
		f, b = torch.chunk(emb, 2, dim=0)
		neg = f @ b.permute(1,0)
		pos = f @ f.permute(1,0) + b @ b.permute(1,0)
		return self.softmax(pos, neg)

	def infonce_v2(self, emb):
		#V0:info-nce
		f, b = torch.chunk(emb, 2, dim=0)
		neg = f @ b.permute(1,0)
		pos1 = f @ f.permute(1,0)
		pos2 = b @ b.permute(1,0)
		return self.softmax(pos1, neg) + self.softmax(pos2, neg)

	@staticmethod
	def rand(feat, pred, mask):
		return MLPSampler(mode='rand').select(feat, pred, mask)
	@staticmethod
	def half(feat, pred, mask):
		return MLPSampler(mode='half').select(feat, pred, mask)
	@staticmethod
	def part(feat, pred, mask):
		return MLPSampler(mode='part').select(feat, pred, mask)
	@staticmethod
	def hard(feat, pred, mask):
		return MLPSampler(mode='hard').select(feat, pred, mask)

	def select(self, feat, pred, mask, ksize=5):
		# assert mode in ['hard', 'semi', 'hazy', 'edge'], 'sample selection mode wrong!'
		assert feat.shape[-2:]==mask.shape[-2:], 'shape of feat & mask donot match!'
		assert feat.shape[-2:]==pred.shape[-2:], 'shape of feat & pred donot match!'
		# reshape embeddings
		feat = feat.permute(0,2,3,1).reshape(-1, feat.shape[1])
		# feat = F.interpolate(feat, size=mask.shape[-2:], mode='bilinear', align_corners=True)
		# feat = F.normalize(feat, p=2, dim=-1)
		mask = mask.round()
		# back = (F.max_pool2d(mask, (ksize, ksize), 1, ksize//2) - mask).round()
		back = (1-mask).round()
		# assert back.sum()>0, 'back has no pixels!'
		# assert mask.sum()>0, 'mask has no pixels!'
		# print('back', back.sum().item(), back.sum().item()/back.numel())
		# print('fgd:', mask.sum().item(), mask.sum().item()/mask.numel())
		# print(feat.shape, pred.shape, mask.shape)

		fh, fl = self.func(feat,   pred, mask, top=self.top, low=self.low)
		bh, bl = self.func(feat, 1-pred, back, top=self.top, low=self.low)
		# print('mlp_sample:', fh.shape, fl.shape, bh.shape, bl.shape)
		# return [fh, fl, bh, bl]
		# print(mode)
		return torch.cat([fh, fl, bh, bl], dim=0)

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
'''

	def infonce(self, emb):
		#V0:info-nce
		f, b = torch.chunk(emb, 2, dim=0)
		neg = f @ b.permute(1,0)
		pos = f @ f.permute(1,0) + b @ b.permute(1,0)
		
		# #V1:split-nce
		# f1,f2, b1,b2 = torch.chunk(emb, 4, dim=0)
		# pos = f1 @ f2.permute(1,0) + b1 @ b2.permute(1,0)
		# neg = f1 @ b1.permute(1,0) + f1 @ b2.permute(1,0) + f2 @ b1.permute(1,0) + f2 @ b2.permute(1,0)

		# #V2:focal-nce
		# seq = (neg.detach()-neg.min().item())/neg.max().item()
		# neg = neg * (1.5 - seq)**2

		pos = torch.exp(pos/self.temp)
		neg = torch.exp(neg/self.temp)

		los = -torch.log(pos / (pos + neg + 1e-5)).mean()
		return los
		'''


def regular(emb, temp=1):
	# f, b = torch.chunk(emb, 2, dim=0)
	# neg = f @ b.permute(1,0)
	# pos = f @ f.permute(1,0) + b @ b.permute(1,0)
	
	f1,f2, b1,b2 = torch.chunk(emb, 4, dim=0)
	pos = f1 @ f2.permute(1,0) + b1 @ b2.permute(1,0)
	neg = f1 @ b1.permute(1,0) + f1 @ b2.permute(1,0) + f2 @ b1.permute(1,0) + f2 @ b2.permute(1,0)

	print('pos:', pos.min().item(), pos.max().item())
	print('neg:', neg.min().item(), neg.max().item())
	seq = (neg.detach()-neg.min().item())/neg.max().item()
	neg = neg * (1.5 - seq)**2
	print('neg:', neg.min().item(), neg.max().item())

	# print(pos.shape, neg.shape)
	pos = torch.exp(pos/ temp)
	neg = torch.exp(neg/ temp)

	los = -torch.log(pos / (pos + neg + 1e-5)).mean()
	return los


if __name__ == '__main__':

	feat = torch.rand(7,32,64,72)
	pred = torch.rand(7,1, 64,72)
	true = torch.rand(7,1, 64,72).round()
	feat = F.normalize(feat, p=2, dim=1)

	sample1 = MLPSampler(top=1, low=4)
	emb = sample1.select(feat, pred, true)
	print(emb.shape)
	print(regular(emb).item())
	# fgd = torch.cat([emb[0], emb[1]], dim=0)
	# bgd = torch.cat([emb[2], emb[3]], dim=0)
	# print(fgd.shape, bgd.shape)

	# sample2 = MLPSampler(top=2, low=4, mode='part')
	# emb = sample2.select(feat, pred, true)
	# print(emb.shape)

	# sample3 = MLPSampler(top=2, low=4, mode='rand')
	# emb = sample3.select(feat, pred, true)
	# print(emb.shape)

	# emb = MLPSampler.rand(feat, pred, true)
	# print(emb.shape)

	# emb = MLPSampler.half(feat, pred, true)
	# print(emb.shape)

	emb = MLPSampler.hard(feat, pred, true)
	print(emb.shape)

	# plot(emb)
	# plt.show()