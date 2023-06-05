import torch
from torch import nn
import torch.nn.functional as F


def compute_kl_loss(p, q, pad_mask=None):
	
	p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='sum')
	q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='sum')
	
	# # pad_mask is for seq-level tasks
	# if pad_mask is not None:
	# 	p_loss.masked_fill_(pad_mask, 0.)
	# 	q_loss.masked_fill_(pad_mask, 0.)

	# # You can choose whether to use function "sum" and "mean" depending on your task
	# p_loss = p_loss.sum()
	# q_loss = q_loss.sum()

	loss = (p_loss + q_loss) / 2
	return loss



'''
https://github.com/dropreg/R-Drop
R-drop is a simple yet very effective regularization method built upon dropout, 
by minimizing the bidirectional KL-divergence of the output distributions 
of any pair of sub models sampled from dropout in model training.

[1]X. Liang et al., “R-Drop: Regularized Dropout for Neural Networks,” 
arXiv:2106.14448 [cs], Jun. 2021, Accessed: Jul. 21, 2021. [Online]. 
Available: http://arxiv.org/abs/2106.14448
'''
if __name__ == '__main__':
			
	# define your task model, which outputs the classifier logits
	# model = TaskModel()
	model = nn.Sequential(
		nn.Conv2d(3,3,3,1,1),
		nn.Dropout2d(),
		nn.Conv2d(3,1,3,1,1),
		nn.Dropout2d(),
		nn.Sigmoid()
	)
	x = torch.rand(5,3,64,64)

	# keep dropout and forward twice
	logits = model(x)
	label = torch.rand_like(logits).round()

	logits2 = model(x)

	# cross entropy loss for classifier
	ce_loss = 0.5 * (F.mse_loss(logits, label) + F.mse_loss(logits2, label))

	kl_loss = compute_kl_loss(logits, logits2)

	# carefully choose hyper-parameters
	loss = ce_loss + kl_loss

	print(loss.item(), '=', ce_loss.item(), '+', kl_loss.item())