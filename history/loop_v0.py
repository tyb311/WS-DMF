#start#
import time, tqdm#, socket
from torchvision.transforms import functional as f

class KerasBackend(object):
	bests = {'auc':0, 'iou':0, 'f1s':0}

	path_minlos = 'checkpoint_minloss.pt'
	path_metric = 'checkpoint_metrics.tar'
	paths = dict()
	logTxt = []
	isParallel = False
	def __init__(self, gpu_first=True, gpu_name='cuda:0', param_init=True, **args):
		super(KerasBackend, self).__init__()
		# print('*'*32,'device')
		torch.manual_seed(311)
		self.device = torch.device('cpu')
		self.param_init = param_init
		if torch.cuda.is_available() and gpu_first:
			self.device = torch.device(gpu_name)  
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
		if not os.path.exists(self.name_folder):
			os.mkdir(self.name_folder)
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
		timeStr = time.strftime("%m%d", time.localtime())
		self.name_folder = '{}{}-{}-{}'.format(timeStr, dataset.dbname, self.model.__name__, losStr)
		dataset.expCross = hasattr(dataset, 'expCross') and dataset.expCross
		if dataset.expCross: 
			self.path_metric = '{}/{}xcp.tar'.format(self.name_folder, dataset.dbname)
			self.path_minlos = '{}/{}xlos.pt'.format(self.name_folder, dataset.dbname)
		else:
			self.path_metric = '{}/{}_cp.tar'.format(self.name_folder, dataset.dbname)
			self.path_minlos = '{}/{}_los.pt'.format(self.name_folder, dataset.dbname)
		print('Folder for experiment:', self.name_folder)

		name_pt = dataset.dbname+'x' if dataset.expCross else dataset.dbname
		# print('Exec:', self.name_folder)
		for key in self.bests.keys():
			self.paths[key] = '{}/{}-{}.pt'.format(self.name_folder, name_pt, key)

	def compile(self, dataset, loss=['Fo', 'Io'], lrs=[0.01, 0.01], expContinue=True, **args): 
		#设置路径
		self.dataset = dataset
		self.init_folders(dataset, ''.join(loss))

		# 参数设置：反向传播、断点训练 
		self.gradUtil = GradUtil(model=self.model, losses=loss, lrs=lrs, expContinue=expContinue, root=self.name_folder)
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
	def __init__(self, model, evalMetric=True, evalEpochs=3, **args):
		super(KerasTorch, self).__init__(**args)
		self.model = model
		self.evalEpochs = evalEpochs
		self.evalMetric = evalMetric   

	def desc(self, key='my'):#, self.scheduler.get_lr()[0] 
		# print('Learing Rate:', self.optimizer.param_groups[0]['lr'])
		for n,m in self.model.named_parameters():
			if n.__contains__(key):
				print(n,m.detach().cpu().numpy())

	def fit(self, epochs=196):#现行验证，意义不大，把所有权重都验证要花不少时间
		self.stop_counter = 0
		self.stop_training = False            
		print('*'*32,'fitting:'+self.name_folder) 
		# self.desc()
		for i in range(epochs):
			# 训练
			lossItem = self.train()
			logStr = '{:03}$ los={}'.format(i, lossItem)
			print('\r'+logStr)
			self.logTxt.append(logStr)
			self.gradUtil.scheduler(i)
			# 验证
			if self.evalMetric and self.gradUtil.isLrLowest(thresh=1e-4):
				# if i>1 and i%self.evalEpochs==0:
				scores, lossItem = self.val()
				logStr = '{:03}$ auc={:.4f} & iou={:.4f} & iou={:.4f}'
				logStr = logStr.format(i, scores['auc'],scores['iou'],scores['f1s'])
				print('\r'+logStr)
				self.logTxt.append(logStr)
				self.callBackEarlyStopping(lossItem, i)
				self.callBackModelCheckPoint(scores)
			else:
				self.callBackEarlyStopping(lossItem)#eye does not use this line
			# 早停
			if self.stop_training==True:
				print('Stop Training!!!')
				break
		self.desc()
		with open(self.name_folder + '/logs.txt', 'w') as f:
			f.write('\n'.join(self.logTxt))
		if self.evalMetric:
			print(self.bests)
	
	useLossVAE = False
	def train(self):
		torch.set_grad_enabled(True)
		self.model.train()     
		lossItem = 0
		tbar = tqdm.tqdm(self.dataset.trainSet())
		for i, imgs in enumerate(tbar):
			(img, lab, fov, aux) = self.dataset.parse(imgs, mix=True)#cpu
			
			if not (isinstance(img, dict) or isinstance(img, list)):
				img = img.to(self.device)

			out = self.model(img)#*fov.to(self.device)
			_lossItem = self.gradUtil.backward_seg(out, lab.to(self.device), fov, self.model, requires_grad=True) 
			lossItem += _lossItem
			del out, lab, fov, aux   
			# print('\r{:03}$ los={:.3f}'.format(i, _lossItem), end='')
			tbar.set_description('{:03}$ los={:.3f}'.format(i, _lossItem))
		return lossItem

	def val(self):
		torch.set_grad_enabled(False)
		self.model.eval()        
		sum_auc = 0
		sum_iou = 0
		sum_f1s = 0
		sum_los = 0
		dataloader = self.dataset.valSet()
		for i, imgs in enumerate(dataloader):
			(img, lab, fov, aux) = self.dataset.parse(imgs, mix=False) 
			pred = self.predict(img, fov)
			loss = self.gradUtil.backward_seg(pred, lab.to(self.device), fov, self.model, requires_grad=False) 
			sum_los += loss
			true = lab.view(-1).numpy().astype(np.float32).reshape(-1)
			pred = pred.cpu().view(-1).numpy().astype(np.float32).reshape(-1)
			if fov is not None:
				fov = fov.view(-1).numpy().astype(np.bool)  
				true, pred = true[fov], pred[fov]
				
			true = np.round(true)
			auc = metrics.roc_auc_score(true, pred)
			sum_auc += auc

			pred = np.round(np.clip(pred, 1e-6, 1-1e-6))
			iou = metrics.jaccard_score(true, pred)
			sum_iou += iou
			f1s = metrics.f1_score(true, pred, average='binary')
			sum_f1s += f1s
			print('\r{:03}$ auc={:.4f} & iou={:.4f} & f1s={:.4f}'.format(i, auc, iou, f1s), end='')
		num = len(dataloader)#i+1#
		los = sum_los/num
		scores = {'auc':sum_auc/num, 'iou':sum_iou/num, 'f1s':sum_f1s/num}
		return scores, los

	def predict(self, img, fov=None, aux=None, mode=None, *args):
		self.model.eval()
		torch.set_grad_enabled(False)
		# with torch.no_grad():  
		if not (isinstance(img, dict) or isinstance(img, list)):
			img = img.to(self.device)      
		pred= self.model(img)
		if isinstance(pred, list) or isinstance(pred, tuple):
			pred = pred[0]
		pred = pred.detach()
		# pred = pred*fov if fov is not None else pred
		return pred.clamp(0, 1)

	def test(self, testset, flagSave=False, key='los', inc='', *args):
		print('*'*32,'testing:',self.name_folder)
		torch.set_grad_enabled(False)
		if not self.load_weights(key):
			# print('there is no weight named:', key)
			return 

		# 设置测试时增强
		forward = self.predict
		name = key + inc
		
		# 计算测试分数
		csv_score = '{}/{}_{}'.format(self.name_folder, self.dataset.dbname, testset.dbname)
		folder_pred = '{}_{}'.format(csv_score, key)
		timeSum = 0

		self.scoreKit = ScoreScikit(csv_score)
		for i, imgs in enumerate(testset.testSet()):
			(img, lab, fov, aux) = testset.parse(imgs)
			st = time.time()
			pred = forward(img, fov, aux)
			timeSum += time.time()-st

			############# 转为图片
			pred = pred.squeeze().cpu().numpy()
			lab = lab.squeeze().numpy()
			pred, lab = testset.post(pred, lab, i)
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

			fov = (fov.squeeze().numpy()>.5).astype(np.bool) 
			self.scoreKit.score(true[fov], pred[fov])

		self.scoreKit.end(model_name=name)
		print('Mean inference time:', timeSum/(i+1))
#end#