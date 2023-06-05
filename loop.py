# 常用资源库
import pandas as pd
import numpy as np
EPS = 1e-9#
import os,glob,numbers
# 图像处理
import math,cv2,random
from PIL import Image, ImageFile, ImageOps, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True
# 图像显示
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

import torch
import torch.nn as nn
import torch.nn.functional as F

#start#
import time, tqdm, wandb#, socket
from torchvision.transforms import functional as f

class KerasBackend(object):
	bests = {'auc':0, 'iou':0, 'f1s':0, 'a':0}

	path_minlos = 'checkpoint_minloss.pt'
	path_metric = 'checkpoint_metrics.tar'
	paths = dict()
	logTxt = []
	isParallel = False
	def __init__(self, args, **kargs):
		super(KerasBackend, self).__init__()
		self.args = args
		# print('*'*32,'device')
		torch.manual_seed(311)
		self.device = torch.device('cpu')
		if torch.cuda.is_available():
			self.device = torch.device('cuda:0')  
			torch.cuda.empty_cache()
			torch.cuda.manual_seed_all(311)
			torch.backends.cudnn.benchmark = False
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.enabled = True
			# Benchmark模式会提升计算速度，但计算中随机性使得每次网络前馈结果略有差异，
			# deterministic避免这种波动, 设置为False可以牺牲GPU提升精度

			current_device = torch.cuda.current_device()
			print(self.device, torch.cuda.get_device_name(current_device))
			for i in range(torch.cuda.device_count()):
				print("    {}:".format(i), torch.cuda.get_device_name(i))
		
	def save_weights(self, path):
		if not os.path.exists(self.root):
			os.mkdir(self.root)
		if self.isParallel:
			torch.save(self.model.module.state_dict(), path)
		else:
			torch.save(self.model.state_dict(), path)
		# print('save weigts to path:{}'.format(path))
	
	def load_weights(self, mode, desc=True):
		path = self.paths.get(mode, mode)#返回完全路径或者mode
		if mode=='los':
			path = self.path_minlos
		try:
			pt = torch.load(path, map_location=self.device)
			self.model.load_state_dict(pt, strict=False)#
			if self.isParallel:
				self.model = self.model.module
			if desc:print('Load from:', path)
			return True
		except Exception as e:
			print('Load wrong:', path)
			return False

	def init_weights(self):
		print('*'*32, 'Initial Weights--Ing!')
		for m in self.model.modules():
			if isinstance(m, nn.Conv2d) and  m.weight.requires_grad:
				torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif isinstance(m, nn.Linear) and  m.weight.requires_grad:
				torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif isinstance(m, nn.BatchNorm2d) and  m.weight.requires_grad:
				torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
				torch.nn.init.constant_(m.bias.data, 0.0)

	def init_folders(self, dataset, losStr):
		timeStr = time.strftime("%m%d%H", time.localtime())
		if self.args.root=='':
			self.root = '{}{}-{}-{}'.format(timeStr, dataset.dbname, self.model.__name__, losStr)
			if self.args.bug:
				self.root = 'BUG'+self.root
		else:
			self.root = self.args.root

		dataset.expCross = hasattr(dataset, 'expCross') and dataset.expCross
		if dataset.expCross: 
			self.path_metric = '{}/{}xcp.tar'.format(self.root, dataset.dbname)
			self.path_minlos = '{}/{}xlos.pt'.format(self.root, dataset.dbname)
		else:
			self.path_metric = '{}/{}_cp.tar'.format(self.root, dataset.dbname)
			self.path_minlos = '{}/{}_los.pt'.format(self.root, dataset.dbname)
		print('Folder for experiment:', self.root)

		name_pt = dataset.dbname+'x' if dataset.expCross else dataset.dbname
		# print('Exec:', self.root)
		for key in self.bests.keys():
			self.paths[key] = '{}/{}-{}.pt'.format(self.root, name_pt, key)

	def compile(self, dataset, loss='fr', lr=0.01, **args): 
		#设置路径
		self.dataset = dataset
		self.init_folders(dataset, ''.join(loss))

		# 参数设置：反向传播、断点训练 
		self.gradUtil = GradUtil(model=self.model, loss=loss, lr=lr, root=self.root)
		if not self.load_weights(self.path_minlos):
			self.init_weights()
		print('Params total(KB):',sum(p.numel() for p in self.model.parameters()))#//245
		print('Params train(KB):',sum(p.numel() for p in self.model.parameters() if p.requires_grad))
			
		if self.isParallel:
			print('*'*32, 'Model Parallel')
			self.model = nn.DataParallel(self.model)  #, device_ids=[0,1,2,3]
			# self.model = nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))  
			# torch.cuda.set_device(self.device)
			# self.device = torch.device('cuda:0')
			self.model.to(self.device)
		else:
			print('*'*32, 'Model Serial')
			self.model = self.model.to(self.device) 

		try:
			self.bests = torch.load(self.path_metric)
			print('Metric Check point:', self.bests)
		except:
			print('Metric Check point none!') 
		
		self.gradUtil.criterion = self.gradUtil.criterion.to(self.device)
		
	def callBackModelCheckPoint(self, scores, lossItem=1e9):
		logStr = '\t'
		for mode in scores.keys():
			if scores[mode]>self.bests[mode]:
				logStr += '{}:{:6.4f}->{:6.4f},'.format(mode, self.bests[mode], scores[mode])
				self.bests[mode] = scores[mode]
				self.save_weights(self.paths[mode])   
		print(logStr)
		self.logTxt.append(logStr)
		torch.save(self.bests, self.path_metric)
		
	stop_counter=0
	stop_training = False
	best_loss = 9999
	isBestLoss = False
	def callBackEarlyStopping(self, los, epoch=0, patience=18):
		if los<self.best_loss:
			print('\tlos={:6.4f}->{:6.4f}'.format(self.best_loss, los))
			self.best_loss = los
			self.stop_counter=0
			# self.save_weights(self.path_minlos)
			self.isBestLoss = True
		else:
			print('\tlos={:6.4f}'.format(los))
			self.stop_counter+=1
			if self.stop_counter>patience and self.gradUtil.isLrLowest(1e-5) and epoch>169:
				self.stop_training = True
				print('EarlyStopp after:', patience)
	
		if self.isBestLoss:
			self.isBestLoss = False
			self.save_weights(self.path_minlos)

class KerasTorch(KerasBackend):
	evalEpochs = 3
	evalMetric = True
	evalEpochs=3
	def __init__(self, model, **kargs):
		super(KerasTorch, self).__init__(**kargs)
		self.model = model

	def desc(self, key='my'):#, self.scheduler.get_lr()[0] 
		# print('Learing Rate:', self.optimizer.param_groups[0]['lr'])
		for n,m in self.model.named_parameters():
			if n.__contains__(key):
				print(n,m.detach().cpu().numpy())

	def plot_emb(self, epoch=0):
		self.dataset.testSet()
		self.model.eval()
		with torch.no_grad():
			imgs = self.dataset.__getitem__(0)
			(img, lab, fov, aux) = self.dataset.parse(imgs)
			lab = lab.to(self.device)
			fov = fov.to(self.device)
			# print(img.shape)
			self.model(img.to(self.device))
			self.model.regular(sampler=self.sampler, lab=lab, fov=fov)
			if len(self.model.feat.shape)==2:
				plot4(emb=self.model.feat, path_save=self.root+f'/feat{epoch}.png')
			else:
				plot4(emb=self.model.proj, path_save=self.root+f'/proj{epoch}.png')
			# print('projection:', out.shape, self.model.feats.shape, lab.shape)
			# print('projection:', out.shape, out.min().item(), out.max().item())
			# embs = self.model.proj.clone().detach()
			# emb_half = self.sampler.half(embs, self.model.pred.detach(), lab, fov)
			# plot4(emb=emb_half, path_save=self.root+f'/emb_half{i}.png')
			# emb_rand = self.sampler.rand(embs, self.model.pred.detach(), lab, fov)
			# plot2(emb=emb_rand, path_save=self.root+f'/emb_rand{i}.png')

	def fit(self, epochs=196):#现行验证，意义不大，把所有权重都验证要花不少时间
		self.stop_counter = 0
		self.stop_training = False            
		print('\n', '*'*32,'fitting:'+self.root) 
		# self.desc()
		time_fit_begin = time.time()
		for i in range(epochs):
			time_stamp = time.time()
			
			# 投影(想象可视化的时候选点策略选hard negatives 还是hazy negatives???)
			# if i%20==0 and self.args.sss!='':	#从数据集中取一张图片用来画图
			# 	self.plot_emb(epoch=i)

			# 训练
			lossItem = self.train()
			if args.board:
				wandb.log({"loss": lossItem, "LR":self.gradUtil.optimizer.param_groups[0]['lr']})

			logStr = '{:03}$ los={}'.format(i, lossItem)
			print('\r'+logStr)
			self.logTxt.append(logStr)
			self.gradUtil.update_scheduler(i)
			# 验证
			if self.evalMetric and self.gradUtil.isLrLowest(thresh=1e-4):
				# if i>1 and i%self.evalEpochs==0:
				scores, lossItem = self.val()
				logStr = '{:03}$ auc={:.4f} & iou={:.4f} & f1s={:.4f}'
				logStr = logStr.format(i, scores['auc'],scores['iou'],scores['f1s'])
				print('\r'+logStr)
				self.logTxt.append(logStr)
				self.callBackEarlyStopping(lossItem, i)
				self.callBackModelCheckPoint(scores)
				if args.board:
					wandb.log(scores)
			else:
				self.callBackEarlyStopping(lossItem)#eye does not use this line
			# 早停
			if self.stop_training==True:
				print('Stop Training!!!')
				break
			
			time_epoch = time.time() - time_stamp
			print('{:03}* {:.2f} mins, left {:.2f} hours to run'.format(i, time_epoch/60, time_epoch/60/60*(epochs-i)))
			if self.args.bug and i>2:
				break
		self.desc()
		if self.evalMetric:
			print(self.bests)
		self.logTxt.append(str(self.bests))
		with open(self.root + '/logs.txt', 'w') as f:
			f.write('\n'.join(self.logTxt))
		if hasattr(self.model, 'tmp'):
			tensorboard_logs(self.model.tmp, root=self.root)
	
		logTime = '\nRunning {:.2f} hours for {} epochs!'.format((time.time() - time_fit_begin)/60/60, epochs)
		print(logTime)

	def train(self):
		torch.set_grad_enabled(True)
		self.model.train()     
		lossItem = 0
		tbar = tqdm.tqdm(self.dataset.trainSet(bs=self.args.bs))
		for i, imgs in enumerate(tbar):
			(img, lab, fov, aux) = self.dataset.parse(imgs)#cpu
			lab = lab.to(self.device)
			fov = fov.to(self.device)
			aux = aux.to(self.device)
			if not (isinstance(img, dict) or isinstance(img, list)):
				img = img.to(self.device)

			losInit = []
			# print(img.shape)
			out = self.model(img)
			if self.args.sss!='':
				los = self.model.regular(sampler=self.sampler, lab=lab, fov=fov) * self.args.coff_cl
				losInit.append(los)
			if self.args.ct:#hasattr(self.model, 'constraint'):
				los1, los2 = self.model.constraint(lab=lab, fov=fov, aux=aux, fun=self.loss_ct)
				los = (los1 * self.args.coff_ds + los2) * self.args.coff_ct
				losInit.append(los)

			if self.args.coff_ce!=0:
				losStd = self.model.encoder.fcn.regular_bce()*self.args.coff_ce
				losInit.append(losStd)

			if self.args.coff_rot!=0:
				los = self.model.encoder.fcn.regular_rot() * self.args.coff_rot
				losInit.append(los)

			# print('backward:', out.shape, lab.shape, fov.shape)
			_lossItem, losStr = self.gradUtil.backward_seg(out, lab, fov, self.model, requires_grad=True, losInit=losInit) 
			lossItem += _lossItem
			del out, lab, fov, aux   
			# print('\r{:03}$ los={:.3f}'.format(i, _lossItem), end='')
			tbar.set_description('{:03}$ {:.3f}={}'.format(i, _lossItem, losStr))
			if self.args.bug and i>3:
				break
		return lossItem

	def predict(self, img, *args):
		self.model.eval()
		torch.set_grad_enabled(False)
		# with torch.no_grad():  
		if not (isinstance(img, dict) or isinstance(img, list)):
			img = img.to(self.device)      
		pred = self.model(img)#*fov.to(self.device)
		if isinstance(pred, dict):
			pred = pred['pred']
		if isinstance(pred, (list, tuple)):
			pred = pred[0]
		pred = pred.detach()
		# pred = pred*fov if fov is not None else pred
		return pred.clamp(0, 1)

	def testaug(self, x0):#Test Time Augment(TTA)， 测试时增强
		torch.set_grad_enabled(False)
		self.model.eval()
		# with torch.no_grad():      
		# before predict  
		x1 = torch.flip(x0, dims=(2,))
		x2 = torch.flip(x0, dims=(3,))
		x3 = torch.transpose(x0, 2, 3)
		x4 = torch.transpose(torch.flip(x0, dims=(2,)), 2, 3)
		x5 = torch.transpose(torch.flip(x0, dims=(3,)), 2, 3)
		# predict
		x0 = self.model(x0.to(self.device)).detach().cpu()
		x1 = self.model(x1.to(self.device)).detach().cpu()
		x2 = self.model(x2.to(self.device)).detach().cpu()
		x3 = self.model(x3.to(self.device)).detach().cpu()
		x4 = self.model(x4.to(self.device)).detach().cpu()
		x5 = self.model(x5.to(self.device)).detach().cpu()
		# after predict
		x1 = torch.flip(x1, dims=(2,))
		x2 = torch.flip(x2, dims=(3,))
		x3 = torch.transpose(x3, 2, 3)
		x4 = torch.flip(torch.transpose(x4, 2, 3), dims=(2,))
		x5 = torch.flip(torch.transpose(x5, 2, 3), dims=(3,))
		
		pred = (x0+x1+x2+x3+x4+x5)/6
		del x0,x1,x2,x3,x4,x5
		return pred.clamp(0, 1)#.cpu()

	def val(self):
		torch.set_grad_enabled(False)
		self.model.eval()        
		sum_auc = 0
		sum_iou = 0
		sum_f1s = 0
		sum_los = 0
		sum_f = 0
		sum_c = 0
		sum_a = 0
		sum_l = 0
		fcal = FCAL()
		dataloader = self.dataset.valSet()
		for i, imgs in enumerate(dataloader):
			(img, lab, fov, aux) = self.dataset.parse(imgs) 
			pred = self.predict(img)
			loss = self.gradUtil.backward_seg(pred, lab.to(self.device), fov, self.model, requires_grad=False)[0] 
			sum_los += loss
			true = lab.squeeze().numpy().astype(np.float32)
			pred = pred.cpu().squeeze().numpy().astype(np.float32)

			f,c,a,l = fcal.forward(true, pred)
			sum_a += a

			true = true.reshape(-1)
			pred = pred.reshape(-1)
			if fov is not None:
				fov = fov.view(-1).numpy().astype(np.bool_)  
				true, pred = true[fov], pred[fov]
				
			true = np.round(true)
			auc = metrics.roc_auc_score(true, pred)
			sum_auc += auc

			pred = np.round(np.clip(pred, 1e-6, 1-1e-6))
			iou = metrics.jaccard_score(true, pred)
			sum_iou += iou
			f1s = metrics.f1_score(true, pred, average='binary')
			sum_f1s += f1s
			print('\r{:03}$ auc={:.4f} & iou={:.4f} & f1s={:.4f} & a={:.4f}'.format(i, auc, iou, f1s, a), end='')
			if self.args.bug and i>2:
				break
		num = len(dataloader)#i+1#
		los = sum_los/num
		scores = {'auc':sum_auc/num, 'iou':sum_iou/num, 'f1s':sum_f1s/num, 'a':sum_a/num}
		return scores, los

	def test(self, testset, flagSave=False, key='los', inc='', tta=False, *args):
		print('\n', '*'*32,'testing:',self.root)
		torch.set_grad_enabled(False)
		if not self.load_weights(key):
			print('there is no weight named:', key)
			# return 
		name = key + inc
		
		# 计算测试分数
		csv_score = '{}/{}_{}'.format(self.root, self.dataset.dbname, testset.dbname)
		folder_pred = '{}_{}{}'.format(csv_score, key, '_tta' if tta else '')
		timeSum = 0

		self.score_pred = ScoreScikit(csv_score)
		# self.score_otsu = ScoreScikit(csv_score)
		for i, imgs in enumerate(testset.testSet()):
			(img, lab, fov, aux) = testset.parse(imgs)
			st = time.time()
			if tta:
				pred = self.testaug(img)
			else:
				pred = self.predict(img)
			timeSum += time.time()-st

			############# 转为图片
			pred, lab, fov = testset.post(pred, lab, fov)
			# print(pred.shape, pred.min().item(), pred.max().item())
			pred = Image.fromarray((pred*255).astype(np.uint8))

			############# 保存图片
			if flagSave or key=='los':
				if not os.path.exists(folder_pred):
					os.mkdir(folder_pred)
				pred.save('{}/{}{:02d}.png'.format(folder_pred, name, i))

			############# 计算得分
			pred = np.asarray(pred).astype(np.float32)/255
			true = np.round(lab)
			# print(i, pred.shape, fov.shape, true.shape)

			fov = (fov>.5).astype(np.bool_) 
			self.score_pred.score(true[fov], pred[fov])
			# self.score_otsu.score(true[fov], pred[fov], otsu=True)

			if self.args.bug and i>2:
				break
		self.score_pred.end(model_name=name + ('_tta' if tta else ''))
		# self.score_otsu.end(model_name=name+'_otsu')
		print('Mean inference time:', timeSum/(i+1))
#end#