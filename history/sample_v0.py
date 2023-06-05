import torch
import torch.nn as nn
import torch.nn.functional as F



#start#
def points_selection_hard(feats, prob, mask, k=256):#point selection by ranking
	assert len(feats.shape)==2, 'feats should contains N*L two dims!'
	L = feats.shape[-1]
	# print(feats.shape, mask.shape)
	feat = feats[mask.view(-1,1).repeat(1, L)>.5].view(-1, L)
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
	
def points_selection_semi(feats, prob, mask, k=256, k1=100):#point selection by prob
	# [1]T. T. Cai, J. Frankle, D. J. Schwab, and A. S. Morcos, 
	# “Are all negatives created equal in contrastive instance discrimination?,” 
	# arXiv:2010.06682 [cs, eess], Oct. 2020, Accessed: Jun. 24, 2021. [Online].
	# 只有top1%-5%的困难负样本是最优效果的，top1%甚至有害，这个top1%的度怎么把握？
	assert len(feats.shape)==2, 'feats should contains N*L two dims!'
	L = feats.shape[-1]
	# print(feats.shape, mask.shape)
	# 取前景像素
	feat = feats[mask.view(-1,1).repeat(1, L)>.5].view(-1, L)
	with torch.no_grad():
		prob = prob[mask>.5].view(-1)
		# seq, idx = torch.sort(prob, dim=-1, descending=True)
		# print(seq.shape, idx.shape, seq[0].item(), idx[0].item(), seq[-1].item(), idx[-1].item())
		seq, idx = torch.sort(prob, dim=-1, descending=True)
		k1 = min(128, idx.numel()//k1)#1%

		h = torch.index_select(feat, dim=0, index=idx[k1:k1+k])
		l = torch.index_select(feat, dim=0, index=idx[-k1-k:-k1])
		# print('lh:', l.shape, h.shape)
		# print(prob[idx[k1:k1+k]].view(-1)[:9])
		# print(prob[idx[-k1-k:-k1]].view(-1)[:9])
	return h, l

# 或许负样本只考虑0.5概率左右的嘞，太难的不考虑了
def points_selection_hazy(feats, prob, mask, k=256, margin=0.4):#point selection hazy points
	assert len(feats.shape)==2, 'feats should contains N*L two dims!'
	L = feats.shape[-1]
	# print(feats.shape, mask.shape)
	feat = feats[mask.view(-1,1).repeat(1, L)>.5].view(-1, L)
	with torch.no_grad():
		prob = prob[mask>.5].view(-1)
		# seq, idx = torch.sort(prob, dim=-1, descending=True)
		# print(seq.shape, idx.shape, seq[0].item(), idx[0].item(), seq[-1].item(), idx[-1].item())
		idx = torch.sort(prob, dim=-1, descending=True)[1][:k]
		# print(prob[idx[:9]])
		h = torch.index_select(feat, dim=0, index=idx)

		idx = torch.sort((prob-margin).abs(), dim=-1, descending=True)[1][-k:]
		# print(prob[idx[:9]])
		l = torch.index_select(feat, dim=0, index=idx)
		# print('lh:', l.shape, h.shape)
	return h, l

def points_selection_edge(feats, prob, mask, k=256, margin=0.5):#point selection hazy points
	assert len(feats.shape)==2, 'feats should contains N*L two dims!'
	L = feats.shape[-1]
	# print(feats.shape, mask.shape)
	feat = feats[mask.view(-1,1).repeat(1, L)>.5].view(-1, L)
	with torch.no_grad():
		tensor2 = torch.tensor(2.0, device=feats.device)
		prob = prob[mask>.5].view(-1)
		# 找出概率大于0.5但接近0.5的样本
		idx = torch.sort(torch.where(prob>margin, prob, tensor2), dim=-1, descending=False)[1][:k]
		# print(prob[idx[:k]][:9])
		h = torch.index_select(feat, dim=0, index=idx)

		# 找出概率小于0.5但接近0.5的样本
		idx = torch.sort(torch.where(prob<margin, 1-prob, tensor2), dim=-1, descending=False)[1][:k]
		# print(prob[idx[:k]][:9])
		l = torch.index_select(feat, dim=0, index=idx)
		# print('lh:', l.shape, h.shape)
	return h, l

def mlp_sample_selection(feat, pred, mask, mode='hazy', ksize=5):
	# assert mode in ['hard', 'semi', 'hazy', 'edge'], 'sample selection mode wrong!'
	assert feat.shape[-2:]==mask.shape[-2:], 'shape of feat & mask donot match!'
	assert feat.shape[-2:]==pred.shape[-2:], 'shape of feat & pred donot match!'
	# reshape embeddings
	feat = feat.permute(0,2,3,1).reshape(-1, feat.shape[1])
	# feat = F.interpolate(feat, size=mask.shape[-2:], mode='bilinear', align_corners=True)
	# feat = F.normalize(feat, p=2, dim=-1)
	mask = mask.round()
	back = (F.max_pool2d(mask, (ksize, ksize), 1, ksize//2) - mask).round()
	# assert back.sum()>0, 'back has no pixels!'
	# assert mask.sum()>0, 'mask has no pixels!'
	# print('back', back.sum().item(), back.sum().item()/back.numel())
	# print('fgd:', mask.sum().item(), mask.sum().item()/mask.numel())
	# print(feat.shape, pred.shape, mask.shape)
	if mode=='semi':
		fh, fl = points_selection_semi(feat, pred, mask)
		bh, bl = points_selection_semi(feat, 1-pred, back)
	elif mode=='hazy':
		fh, fl = points_selection_hazy(feat, pred, mask)
		bh, bl = points_selection_hazy(feat, 1-pred, back)
	elif mode=='edge':
		fh, fl = points_selection_edge(feat, pred, mask)
		bh, bl = points_selection_edge(feat, 1-pred, back)
	# if mode=='hard':
	else:
		fh, fl = points_selection_hard(feat, pred, mask)
		bh, bl = points_selection_hard(feat, 1-pred, back)
	# print('mlp_sample:', fh.shape, fl.shape, bh.shape, bl.shape)
	# return [fh, fl, bh, bl]
	# print(mode)
	return torch.cat([fh, fl, bh, bl], dim=0)
#end#

	

if __name__ == '__main__':

	feat = torch.rand(7,32,64,72)
	pred = torch.rand(7,1, 64,72)
	true = torch.rand(7,1, 64,72).round()

	feats = mlp_sample_selection(feat, pred, true, 'semi')
	print(feats.shape)
	# fgd = torch.cat([feats[0], feats[1]], dim=0)
	# bgd = torch.cat([feats[2], feats[3]], dim=0)
	# print(fgd.shape, bgd.shape)

	feats = mlp_sample_selection(feat, pred, true, 'hard')
	print(feats.shape)
	# fgd = torch.cat([feats[0], feats[1]], dim=0)
	# bgd = torch.cat([feats[2], feats[3]], dim=0)
	# print(fgd.shape, bgd.shape)
	
	feats = mlp_sample_selection(feat, pred, true, 'hazy')
	print(feats.shape)
	# fgd = torch.cat([feats[0], feats[1]], dim=0)
	# bgd = torch.cat([feats[2], feats[3]], dim=0)
	# print(fgd.shape, bgd.shape)

	
	feats = mlp_sample_selection(feat, pred, true, 'edge')
	print(feats.shape)