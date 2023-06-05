# -*- encoding:utf-8 -*-
###Code for segcl
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
import socket
import matplotlib as mpl
if 'TAN' not in socket.gethostname():
	print('Run on Server!!!')
	mpl.use('Agg')#服务器绘图

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as f
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def gain(ret, p=1):    #gain_off
	mean = np.mean(ret)
	ret_min = mean-(mean-np.min(ret))*p
	ret_max = mean+(np.max(ret)-mean)*p
	ret = 255*(ret - ret_min)/(ret_max - ret_min)
	ret = np.clip(ret, 0, 255).astype(np.uint8)
	return ret

def arr2img(pic):
	return Image.fromarray(pic.astype(np.uint8))#, mode='L'

def arrs2imgs(pic):
	_pic=dict()
	for key in pic.keys():
		_pic[key] = arr2img(pic[key])
	return _pic

def imgs2arrs(pic):
	_pic=dict()
	for key in pic.keys():
		_pic[key] = np.array(pic[key])
	return _pic

def pil_tran(pic, tran=None):
	if tran is None:
		return pic
	if isinstance(tran, list):
		for t in tran:
			for key in pic.keys():
				pic[key] = pic[key].transpose(t)
	else:
		for key in pic.keys():
			pic[key] = pic[key].transpose(tran)
	return pic

# class Aug4Val(object):
#     number = 8
#     @staticmethod
#     def forward(pic, flag):
#         flag %= Aug4Val.number
#         if flag==0:
#             return pic
#         pic = arrs2imgs(pic)
#         if flag==1:
#             return imgs2arrs(pil_tran(pic, tran=Image.FLIP_LEFT_RIGHT))
#         if flag==2:
#             return imgs2arrs(pil_tran(pic, tran=Image.FLIP_TOP_BOTTOM))
#         if flag==3:
#             return imgs2arrs(pil_tran(pic, tran=Image.ROTATE_180))
#         if flag==4:
#             return imgs2arrs(pil_tran(pic, tran=[Image.TRANSPOSE]))
#         if flag==5:
#             return imgs2arrs(pil_tran(pic, tran=[Image.TRANSPOSE,Image.FLIP_TOP_BOTTOM]))
#         if flag==6:
#             return imgs2arrs(pil_tran(pic, tran=[Image.TRANSPOSE,Image.FLIP_LEFT_RIGHT]))
#         if flag==7:
#             return imgs2arrs(pil_tran(pic, tran=[Image.TRANSPOSE,Image.ROTATE_180]))
class Aug4Val(object):
	number = 4
	@staticmethod
	def forward(pic, flag):
		flag %= Aug4Val.number
		if flag==0:
			return pic
		pic = arrs2imgs(pic)
		if flag==1:
			return imgs2arrs(pil_tran(pic, tran=Image.FLIP_LEFT_RIGHT))
		if flag==2:
			return imgs2arrs(pil_tran(pic, tran=Image.FLIP_TOP_BOTTOM))
		if flag==3:
			return imgs2arrs(pil_tran(pic, tran=Image.ROTATE_180))
		# if flag==4:
		# 	return imgs2arrs(pil_tran(pic, tran=[Image.FLIP_LEFT_RIGHT,Image.FLIP_TOP_BOTTOM]))
		# if flag==5:
		# 	return imgs2arrs(pil_tran(pic, tran=[Image.ROTATE_180,Image.FLIP_TOP_BOTTOM]))
		# if flag==6:
		# 	return imgs2arrs(pil_tran(pic, tran=[Image.ROTATE_180,Image.FLIP_LEFT_RIGHT]))
		# if flag==7:
		# 	return imgs2arrs(pil_tran(pic, tran=[Image.ROTATE_180,Image.FLIP_LEFT_RIGHT,Image.FLIP_TOP_BOTTOM]))

def random_channel(rgb, tran=None):##cv2.COLOR_RGB2HSV,HSV不好#
	if tran is None:
		tran = random.choice([
			cv2.COLOR_RGB2GRAY, cv2.COLOR_RGB2BGR, 
			cv2.COLOR_RGB2XYZ, cv2.COLOR_RGB2LAB,
			cv2.COLOR_RGB2HLS, cv2.COLOR_RGB2LUV,        
			cv2.COLOR_RGB2YCrCb, cv2.COLOR_RGB2YUV,
			])
	# if rgb.shape[-1]!=3:#单通道图片不做变换
	#     return rgb
	rgb = cv2.cvtColor(rgb, tran)
	# if tran==cv2.COLOR_RGB2LAB:
	# 	rgb = cv2.split(rgb)[0]
	# elif tran==cv2.COLOR_RGB2XYZ:
	# 	rgb = cv2.split(rgb)[0]
	# elif tran==cv2.COLOR_RGB2LUV:
	# 	rgb = cv2.split(rgb)[0]
	# elif tran==cv2.COLOR_RGB2HLS:
	# 	rgb = cv2.split(rgb)[1]
	# elif tran==cv2.COLOR_RGB2YCrCb:
	# 	rgb = cv2.split(rgb)[0]
	# elif tran==cv2.COLOR_RGB2YUV:
	# 	rgb = cv2.split(rgb)[0]
	# elif tran==cv2.COLOR_RGB2BGR:
	# 	rgb = cv2.split(rgb)[1]
	
	if tran != cv2.COLOR_RGB2GRAY:
		rgb = random.choice(cv2.split(rgb))#
	return rgb

class Aug4CSA(object):#Color Space Augment
	number = 8
	trans = [
			cv2.COLOR_RGB2GRAY, cv2.COLOR_RGB2BGR, 
			cv2.COLOR_RGB2XYZ, cv2.COLOR_RGB2LAB,
			cv2.COLOR_RGB2HLS, cv2.COLOR_RGB2LUV,        
			cv2.COLOR_RGB2YCrCb, cv2.COLOR_RGB2YUV,
			]
	@staticmethod
	def forward(pic, flag):
		flag %= Aug4CSA.number
		pic['img'] = random_channel(pic['img'], tran=Aug4CSA.trans[flag])
		return pic
	@staticmethod
	def forward_train(pic):  #random channel mixture
		a = random_channel(pic['img'])
		b = random_channel(pic['img'])
		alpha = random.random()
		pic['img'] = (alpha*a + (1-alpha)*b).astype(np.uint8)
		return pic

class EyeSetResource(object):
	size = dict()
	def __init__(self, folder='../eyeset', dbname='drive', loo=None, **args):
		super(EyeSetResource, self).__init__()
		
		if os.path.isdir('/home/tan/datasets/seteye'):
			self.folder = '/home/tan/datasets/seteye'
		elif os.path.isdir('/home/tyb/datasets/seteye'):
			self.folder = '/home/tyb/datasets/seteye'
		else:
			self.folder = '../datasets/seteye'
		# else:
		# 	raise EnvironmentError('No thi root!')
		# self.folder = folder
		self.dbname = dbname

		self.imgs, self.labs, self.fovs, self.skes = self.getDataSet(self.dbname)
		if dbname=='stare' and loo is not None: 
			self.imgs['test'] = [self.imgs['full'][loo]]
			self.imgs['train'] = self.imgs['full'][:loo] + self.imgs['full'][1+loo:]
			self.imgs['val'] = self.imgs['train']
			
			self.labs['test'] = [self.labs['full'][loo]]
			self.labs['train'] = self.labs['full'][:loo] + self.labs['full'][1+loo:]
			self.labs['val'] = self.labs['train']
			
			self.fovs['test'] = [self.fovs['full'][loo]]
			self.fovs['train'] = self.fovs['full'][:loo] + self.fovs['full'][1+loo:]
			self.fovs['val'] = self.fovs['train']
			
			self.skes['test'] = [self.skes['full'][loo]]
			self.skes['train'] = self.skes['full'][:loo] + self.skes['full'][1+loo:]
			self.skes['val'] = self.skes['train']
			print('LOO:', loo, self.imgs['test'])
			print('LOO:', loo, self.labs['test'])
			print('LOO:', loo, self.fovs['test'])
			print('LOO:', loo, self.skes['test'])

		self.lens = {'train':len(self.labs['train']),   'val':len(self.labs['val']),
					 'test':len(self.labs['test']),     'full':len(self.labs['full'])}  
		# print(self.lens)  
		if self.lens['test']>0:
			lab = self.readArr(self.labs['test'][0])
			self.size['raw'] = lab.shape
			h,w = lab.shape
			self.size['pad'] = (math.ceil(h/32)*32, math.ceil(w/32)*32)
			print('size:', self.size)
		else:
			print('dataset has no images!')

		# print('*'*32,'eyeset','*'*32)
		strNum = 'images:{}+{}+{}#{}'.format(self.lens['train'], self.lens['val'], self.lens['test'], self.lens['full'])
		print('{}@{}'.format(self.dbname, strNum))

	def getDataSet(self, dbname):        
		# 测试集
		imgs_test = self.readFolder(dbname, part='test', image='rgb')
		labs_test = self.readFolder(dbname, part='test', image='lab')
		fovs_test = self.readFolder(dbname, part='test', image='fov')
		skes_test = self.readFolder(dbname, part='test', image='ske')
		# 训练集
		imgs_train = self.readFolder(dbname, part='train', image='rgb')
		labs_train = self.readFolder(dbname, part='train', image='lab')
		fovs_train = self.readFolder(dbname, part='train', image='fov')
		skes_train = self.readFolder(dbname, part='train', image='ske')
		# 全集
		imgs_full,labs_full,fovs_full,skes_full = [],[],[],[]
		imgs_full.extend(imgs_train); imgs_full.extend(imgs_test)
		labs_full.extend(labs_train); labs_full.extend(labs_test)
		fovs_full.extend(fovs_train); fovs_full.extend(fovs_test)
		skes_full.extend(skes_train); skes_full.extend(skes_test)

		db_imgs = {'train': imgs_train, 'val':imgs_train, 'test': imgs_test, 'full':imgs_full}
		db_labs = {'train': labs_train, 'val':labs_train, 'test': labs_test, 'full':labs_full}
		db_fovs = {'train': fovs_train, 'val':fovs_train, 'test': fovs_test, 'full':fovs_full}
		db_skes = {'train': skes_train, 'val':skes_train, 'test': skes_test, 'full':skes_full}
		return db_imgs, db_labs, db_fovs, db_skes

	def readFolder(self, dbname, part='train', image='rgb'):
		path = self.folder + '/' + dbname + '/' + part + '_' + image
		imgs = glob.glob(path + '/*.npy')
		imgs.sort()
		return imgs
		
	def readArr(self, image):
		# assert(image.endswith('.npy'), 'not npy file!') 
		return np.load(image) 
	
	def readDict(self, index, exeData):  
		img = self.readArr(self.imgs[exeData][index])
		fov = self.readArr(self.fovs[exeData][index])
		lab = self.readArr(self.labs[exeData][index])
		ske = self.readArr(self.skes[exeData][index])
		if fov.shape[-1]==3:
			fov = cv2.cvtColor(fov, cv2.COLOR_BGR2GRAY)
		return {'img':img, 'lab':lab, 'fov':fov, 'ske':ske}#

import imgaug as ia
import imgaug.augmenters as iaa
IAA_NOISE = iaa.OneOf(children=[# Noise
		iaa.Add((-7, 7), per_channel=True),
		iaa.AddElementwise((-7, 7)),
		iaa.Multiply((0.9, 1.1), per_channel=True),
		iaa.MultiplyElementwise((0.9, 1.1), per_channel=True),

		iaa.AdditiveGaussianNoise(scale=3, per_channel=True),
		iaa.AdditiveLaplaceNoise(scale=3, per_channel=True),
		iaa.AdditivePoissonNoise(lam=5, per_channel=True),

		iaa.SaltAndPepper(0.01, per_channel=True),
		iaa.ImpulseNoise(0.01),
	]
)
IAA_BLEND = iaa.OneOf(children=[# Noise
		# Blend
		iaa.BlendAlpha((0.0, 1.0), foreground=iaa.Add(seed=random.randint(0,9)), background=iaa.Multiply(seed=random.randint(0,9))),
		iaa.BlendAlpha((0.0, 1.0), foreground=iaa.Add(seed=random.randint(0,9)), background=iaa.Add(seed=random.randint(0,9))),
		iaa.BlendAlpha((0.0, 1.0), foreground=iaa.Multiply(seed=random.randint(0,9)), background=iaa.Multiply(seed=random.randint(0,9))),
		iaa.BlendAlpha((0.0, 1.0), foreground=iaa.Multiply(seed=random.randint(0,9)), background=iaa.Add(seed=random.randint(0,9))),

		iaa.BlendAlphaElementwise(.3, iaa.Clouds(), seed=random.randint(0,9), per_channel=True),
		iaa.BlendAlphaElementwise(.5, iaa.AddToBrightness(21), seed=random.randint(0,9), per_channel=True),
		iaa.BlendAlphaElementwise(.5, iaa.AddToHue(21), seed=random.randint(0,9), per_channel=True),
		iaa.BlendAlphaElementwise(.5, iaa.AddToSaturation(21), seed=random.randint(0,9), per_channel=True),

		iaa.BlendAlphaVerticalLinearGradient(iaa.Clouds(), max_value=.5, seed=random.randint(0,9)),
		iaa.BlendAlphaVerticalLinearGradient(iaa.AddToHue((-30, 30)), seed=random.randint(0,9)),
		iaa.BlendAlphaVerticalLinearGradient(iaa.AddToSaturation((-30, 30)), seed=random.randint(0,9)),
		iaa.BlendAlphaVerticalLinearGradient(iaa.AddToBrightness((-30, 30)), seed=random.randint(0,9)),

		iaa.BlendAlphaHorizontalLinearGradient(iaa.Clouds(), max_value=.5, seed=random.randint(0,9)),
		iaa.BlendAlphaHorizontalLinearGradient(iaa.AddToHue((-30, 30)), seed=random.randint(0,9)),
		iaa.BlendAlphaHorizontalLinearGradient(iaa.AddToSaturation((-30, 30)), seed=random.randint(0,9)),
		iaa.BlendAlphaHorizontalLinearGradient(iaa.AddToBrightness((-30, 30)), seed=random.randint(0,9)),

		iaa.BlendAlphaCheckerboard(7, 7, iaa.AddToHue((-30, 30)), seed=random.randint(0,9)),
	]
)
TRANS_NOISE = iaa.Sequential(children=[IAA_NOISE, IAA_BLEND])

from albumentations import (
	# 空间
	RGBShift, ChannelDropout, ChannelShuffle, 
	# 色调
	HueSaturationValue, RandomContrast, RandomBrightness, 
	# 翻转
	Flip, Transpose, RandomRotate90, PadIfNeeded, RandomGridShuffle,
	# 变形
	GridDistortion, ShiftScaleRotate, IAAPiecewiseAffine, OpticalDistortion, ElasticTransform,#IAAPerspective, 
	# 噪声
	IAASharpen, IAAEmboss, GaussNoise, MultiplicativeNoise, #IAAAdditiveGaussianNoise, 
	# 模糊
	MedianBlur, GaussianBlur, #Blur, MotionBlur,
	# 其他
	OneOf, Compose, CropNonEmptyMaskIfExists, CLAHE, RandomGamma
) # 图像变换函数

TRANS_TEST = Compose([CLAHE(p=1), RandomGamma(p=1)])#
TRANS_AAUG = Compose([
	# OneOf([
	# 	IAAPiecewiseAffine(p=1), ElasticTransform(p=1),
	# 	ShiftScaleRotate(p=1, scale_limit=0, rotate_limit=45),
	# 	ShiftScaleRotate(p=1, scale_limit=0.25, rotate_limit=0), 
	# 	OpticalDistortion(p=1, distort_limit=.5, shift_limit=0.5), 
	# 	GridDistortion(p=1, num_steps=1), 
	# 	], p=.7),        
	OneOf([Transpose(p=1), RandomRotate90(p=1), ], p=.7),
	Flip(p=.7), 
	# CLAHE(p=.5), RandomGamma(p=.5),
	# RandomGridShuffle(p=1, grid=(5,5)),
])


from skimage import morphology
from torch.utils.data import DataLoader, Dataset
class EyeSetGenerator(Dataset, EyeSetResource):
	exeNums = {'train':Aug4CSA.number, 'val':Aug4Val.number, 'test':1}
	exeMode = 'train'#train, val, test
	exeData = 'train'#train, test, full

	SIZE_IMAGE = 128
	expCross = False   
	LEN_AUG = 32
	def __init__(self, datasize=128, **args):
		super(EyeSetGenerator, self).__init__(**args)
		self.SIZE_IMAGE = datasize
		self.LEN_AUG = 96 // (datasize//64)**2
		print('SIZE_IMAGE:{} & AUG SIZE:{}'.format(self.SIZE_IMAGE, self.LEN_AUG))
		
	def __len__(self):
		length = self.lens[self.exeData]*self.exeNums[self.exeMode]
		if self.isTrainMode:
			return length*self.LEN_AUG
		return length

	def set_mode(self, mode='train'):
		self.exeMode = mode
		self.exeData = 'full' if self.expCross else mode 
		self.isTrainMode = (mode=='train')
		self.isValMode = (mode=='val')
		self.isTestMode = (mode=='test')
	def trainSet(self, bs=8, data='train'):#pin_memory=True, , shuffle=True
		self.set_mode(mode='train')
		return DataLoader(self, batch_size=bs, pin_memory=True, num_workers=4, shuffle=True)
	def valSet(self, data='val'):
		self.set_mode(mode='val')
		return DataLoader(self, batch_size=1,  pin_memory=True, num_workers=2)
	def testSet(self, data='test'):
		self.set_mode(mode='test')
		return DataLoader(self, batch_size=1,  pin_memory=True, num_workers=2)
	#DataLoader worker (pid(s) 5220) exited unexpectedly, 令numworkers>1就可以啦
	
	# @staticmethod
	def parse(self, pics, cat=True):
		rows, cols = pics['lab'].squeeze().shape[-2:]     
		for key in pics.keys(): 
			# print(key, pics[key].shape)
			pics[key] = pics[key].view(-1,1,rows,cols) 
		return pics['img'], torch.round(pics['lab']), torch.round(pics['fov']), torch.round(pics['ske'])

	def post(self, img, lab, fov):
		if type(img) is not np.ndarray:img = img.squeeze().cpu().numpy()
		if type(lab) is not np.ndarray:lab = lab.squeeze().cpu().numpy()
		if type(fov) is not np.ndarray:fov = fov.squeeze().cpu().numpy()
		img = img * fov
		return img, lab, fov

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
	use_csm = True
	def __getitem__(self, idx, divide=32):
		index = idx % self.lens[self.exeData] 
		pics = self.readDict(index, self.exeData)
		imag = pics['img']# = cv2.cvtColor(pics['img'], cv2.COLOR_RGB2GRAY)

		# pics['aux'] = pics['ske']
		if self.isTrainMode:
			# print(pics['lab'].shape, pics['fov'].shape, pics['aux'].shape)
			mask = np.stack([pics['lab'], pics['fov'], pics['ske']], axis=-1)
			# 裁剪增强
			augCrop = CropNonEmptyMaskIfExists(p=1, height=self.SIZE_IMAGE, width=self.SIZE_IMAGE)
			picaug = augCrop(image=imag, mask=mask)
			imag, mask = picaug['image'], picaug['mask']

			# 随机增强
			# if random.choice([True, False]):
			imag = TRANS_TEST(image=imag)['image']
			# if random.choice([True, False]):
			# imag = TRANS_TEST(image=imag)['image']

			# 添加噪声
			imag = TRANS_NOISE(image=imag)
			# 变换增强
			picaug = TRANS_AAUG(image=imag, mask=mask)
			imag, mask = picaug['image'], picaug['mask']
			
			pics['img'] = imag
			pics['lab'], pics['fov'], pics['ske'] = mask[:,:,0],mask[:,:,1],mask[:,:,2]
			if self.use_csm:
				pics = Aug4CSA.forward_train(pics)
		else:
			# pics['img'] = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
			pics['img'] = TRANS_TEST(image=pics['img'])['image']
			# 图像补齐
			h, w = pics['lab'].shape
			w = int(np.ceil(w / divide)) * divide
			h = int(np.ceil(h / divide)) * divide
			augPad = PadIfNeeded(p=1, min_height=h, min_width=w)
			for key in pics.keys():
				pics[key] = augPad(image=pics[key])['image']

			if self.isValMode:# 验证增强->非测试，则增强
				flag = idx//self.lens[self.exeData]
				# pics = Aug4CSA.forward(pics, flag)
				pics = Aug4Val.forward(pics, flag)
			# elif self.isTestMode:	
				# pics = Aug4CSA.forward_test(pics)

		if pics['img'].shape[-1]==3:#	green or gray
			pics['img'] = cv2.cvtColor(pics['img'], cv2.COLOR_RGB2GRAY)
			# pics['img'] = pics['img'][:,:,1]#莫非灰度图像比绿色通道更好一点？

		# 骨架化
		# skel = morphology.skeletonize((pics['lab']/255.0).round()).astype(np.uint8)
		# pics['ske'] = morphology.dilation(skel, self.kernel)*255
		# pics['ske'] = pics['ske'] | pics['lab']	#与不与，这是个问题

		for key in pics.keys():
			# print(key, pics[key].shape)
			pics[key] = torch.from_numpy(pics[key]).type(torch.float32).div(255)
		return pics

from skimage import morphology
from skimage import measure
class FCAL(object):##   fcal 中a是比较好且稳定的，a>l>f>c
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    def forward(self, RefV, SrcV):
        # % Initialization
        SrcV = SrcV.round()
        RefV = RefV.round()
        # print('FCAL:', SrcV.shape, RefV.shape, self.kernel.shape, self.kernel.shape)
        C,A,L = 1,1,1
        # SrcS = morphology.skeletonize(SrcV, method='zhang').astype(np.uint8)
        # RefS = morphology.skeletonize(RefV, method='zhang').astype(np.uint8)

        RefD = morphology.dilation(RefV, self.kernel)
        SrcD = morphology.dilation(SrcV, self.kernel)
        # print(RefD.min(), RefD.max(), SrcD.min(), SrcD.max())

        # % Calculation of L
        # dilateOverlap = SrcD * RefV + RefD * SrcV
        # interAera = dilateOverlap.sum()
        # L = interAera / (RefV + SrcV).sum()

        # % Calculation of A
        dilateOverlap = SrcD * RefV + RefD * SrcV
        dilateOverlap[dilateOverlap>0] = 1
        Overlap = RefV + SrcV
        Overlap[Overlap>0] = 1
        A = dilateOverlap.sum() / Overlap.sum()

        # # % Calculation of C
        # RefSD = morphology.dilation(RefS, self.kernel)
        # SrcSD = morphology.dilation(SrcS, self.kernel)
        # # print(RefS.min(), RefS.max(), SrcS.min(), SrcS.max())
        # # print(RefSD.min(), RefSD.max(), SrcSD.min(), SrcSD.max())
        # inter = (RefSD * SrcSD).sum()
        # outer = (RefSD + SrcSD).sum() - inter
        # C = inter / outer
        
        # return D, C, A, L
        return C*A*L, C, A, L

class PdRecord(object):
    def __init__(self, name='metrics', index=None, old=True):
        super(PdRecord, self).__init__()
        # assert(isinstance(index, list), 'index must be list')
        self.index = index
        self.name = name
        self.json = self.name + '.json'
        if old and os.path.exists(self.json):
            self.df = pd.read_json(self.json)
        else:
            self.df = pd.DataFrame(index=self.index)

    def save_json(self):
        self.df.to_json(self.json)

    def set_list(self, name, values):
        tag = pd.DataFrame(data=values, index=self.index)  
        # assert(isinstance(name, str), 'name must be str')
        # assert(isinstance(values, list), 'values must be list')
        self.df.__setitem__(name, tag)
        self.save_json()

    def set_item(self, name, value):
        tag = pd.DataFrame(data=[value], index=self.index) 
        self.df.__setitem__(name, tag)
        self.save_json()

    def get_score(self, name):
        if self.df.columns.__contains__(name):
            score = float(self.df.__getitem__(name))
        else:
            score = 0
        print(name + '@score =', score)
        return score

    def desc(self, transpose=False):
        if transpose:
            print(self.df.transpose())
        else:
            print(self.df)

    def end(self, transpose=True):
        df = self.df.transpose() if transpose else self.df
        df.to_csv(self.name + '.csv')
        # print(df)

from sklearn import metrics
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(precision=4)

#['accuracy', 'auc', 'f1_score', 'gmean', 'iou', 'kappa', 'mcc', 'precision', 'sensitivity', 'specificity']
class ScoreScikit(object):
	def __init__(self, json_name='scores', smooth=1e-6):
		super(ScoreScikit, self).__init__()
		self.score_list = []
		self.cnt = 0
		self.json_name = json_name
		self.score_names = ['acc', 'auc', 'f1s', 'gme', 'iou', 'kap', 'mcc', 'pre', 'sen', 'spe']
		
	def calc(self, true, pred, otsu=False, EPS=1e-9):
		true = np.round(true)		
		if np.all(pred==0) or np.all(pred==1):
			auc = 0
		else:
			auc = metrics.roc_auc_score(true, pred)# 计算AUC时对预测不使用阈值
		if otsu:
			pred = (pred*255).astype(np.uint8)
			_,pred = cv2.threshold(pred, 127, 255, cv2.THRESH_OTSU)
			pred = pred/255            
		pred = np.round(pred)    

		(TN, FP),(FN, TP) = metrics.confusion_matrix(true, pred)
		# True Positive (TP) False Negative (FN) False Positive (FP) True Negative (TN)
		acc = (TP+TN)/(TP+TN+FP+FN)#metrics.accuracy_score(true, pred)

		spe = TN/(TN+FP)
		sen = TP/(TP+FN)#recall = sensitivity
		dic = 2*TP/(2*TP+FP+FN)   # F1Score==Dice dic != 2*TP/(2*TP+FP+TN)
		iou = TP/(TP+FP+FN)       # print('IOU:', iou, metrics.jaccard_score(true, pred))
		pc_ = TP/(TP+FP+EPS)      #precison
		g__ = math.sqrt(sen*spe)
		kap = metrics.cohen_kappa_score(true, pred)
		mcc = metrics.matthews_corrcoef(true, pred)  
		f1s = metrics.f1_score(true, pred, average='binary', zero_division=0)
		# f1_ = 2*sen*pc_/(sen+pc_+EPS)
		return np.array([acc, auc, f1s, g__, iou, kap, mcc, pc_, sen, spe]).reshape(1, -1)

	def score(self, true, pred, otsu=False):# 默认黑底白字,0~1 & 0~255
		pred = pred.clip(0,1).astype(np.float32).reshape(-1)
		true = true.clip(0,1).astype(np.float32).reshape(-1)
		# 归一化（省去）
		scores = self.calc(true, pred, otsu=otsu)*10000
		self.cnt+=1
		# print('{:02}'.format(self.cnt), (scores).astype(np.int))
		self.score_list.append(scores)

	def end(self, model_name='net'):
		if self.score_list.__len__()>0:
			scores = np.concatenate(self.score_list, axis=0)
			score_value = (scores.mean(axis=0)).astype(np.int)
			print(model_name, score_value)
			record = PdRecord(name=self.json_name, index=self.score_names)
			record.set_list(model_name, score_value)
			record.end()
	
	def desc(self):
		record = PdRecord(name=self.json_name, index=self.score_names)
		record.desc(transpose=True)

import math
import torch
from torch.optim.optimizer import Optimizer

class RAdamW(Optimizer):
    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
        super(RAdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdamW, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (
                                    1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.mul_(1 - group['lr'] * group['weight_decay'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.mul_(1 - group['lr'] * group['weight_decay'])
                    p_data_fp32.add_(exp_avg, alpha=-step_size * group['lr'])
                    p.data.copy_(p_data_fp32)

        return loss

from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
import math 

class ReduceLR(ReduceLROnPlateau):
	def __init__(self, name='los', **args):
		super(ReduceLR, self).__init__(**args)
		self.name = name
	
	def _reduce_lr(self, epoch):
		for i, param_group in enumerate(self.optimizer.param_groups):
			old_lr = float(param_group['lr'])
			new_lr = max(old_lr * self.factor, self.min_lrs[i])
			if old_lr - new_lr > self.eps:
				param_group['lr'] = new_lr
				print('\t{}: lr down to {:.4e}.'.format(self.name, new_lr))

	def get_lr(self):
		# return self.optimizer.param_groups[0]['lr']
		return [g['lr'] for g in self.optimizer.param_groups]

	def get_last_lr(self):
		# return self.optimizer.param_groups[0]['lr']
		return self.get_lr()[0]

import copy, torch, torchvision, os
def tensorboard_model(model=None, root='./Alogs', nrow=4):
	net = copy.deepcopy(model).cpu()
	for name, item in net.cpu().named_parameters():
		if not item.requires_grad:
			continue
		if len(item.shape)==4 and item.shape[-1]>3 and item.shape[-2]>3:#只显示四维、尺寸大于3的大核
			tag = os.path.join(root, net.__name__+'_'+name+'.png')
			try:
				item = item.view(-1,1,item.shape[-2],item.shape[-1])
				torchvision.utils.save_image(item, tag, nrow=nrow, normalize=True, scale_each=True)
			except:
				print('tensorboard_model-', name, item.shape, item.dtype)

def tensorboard_logs(logs={}, root='./Alogs', nrow=4):
	for name, item in logs.items():
		if isinstance(item, list) or isinstance(item, tuple):
			continue
		tag = os.path.join(root, name+'.png')
		try:
			item = item.view(-1,1,item.shape[-2],item.shape[-1])
			torchvision.utils.save_image(item, tag, nrow=nrow, normalize=True, scale_each=True)
		except:
			print('tensorboard_logs-', name, item.shape, item.dtype)

from torchvision.models import vgg16
class PerceptionLoss(nn.Module):
    def __init__(self, device=torch.device('cpu'), layer=31):
        super(PerceptionLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:layer]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network.to(device)
        self.mse_loss = nn.MSELoss()

    def forward(self, pr, gt):
        pr = torch.cat([pr,pr,pr], dim=1)
        gt = torch.cat([gt,gt,gt], dim=1)
        return self.mse_loss(self.loss_network(pr), self.loss_network(gt))

def gram_matrix(y):
	(b,ch,h,w) = y.size() # 比如1,8,2,2
	features = y.view(b,ch,w*h)  # 得到1,8,4
	features_t = features.transpose(1,2) # 调换第二维和第三维的顺序，即矩阵的转置，得到1,4,8
	gram = features.bmm(features_t)# / (ch*h*w) # bmm()用来做矩阵乘法，及未转置的矩阵乘以转置后的矩阵，得到的就是1,8,8了
    # 由于要对batch中的每一个样本都计算Gram Matrix，因此使用bmm()来计算矩阵乘法，而不是mm()
	return gram

from torchvision.models import vgg16
class TextureLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(TextureLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network.to(device)
        self.mse_loss = nn.MSELoss()

    def forward(self, pr, gt):
        pr = torch.cat([pr,pr,pr], dim=1)
        gt = torch.cat([gt,gt,gt], dim=1)
        perception_loss = self.mse_loss(gram_matrix(self.loss_network(pr)), gram_matrix(self.loss_network(gt)))
        return perception_loss

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
    @staticmethod
    def binary_focal(pr, gt, fov=None, gamma=2, *args):
        return -gt     *torch.log(pr)      *torch.pow(1-pr, gamma)
    def forward(self, pr, gt, fov=None, gamma=2, eps=1e-6, *args):
        pr = torch.clamp(pr, eps, 1-eps)
        loss1 = self.binary_focal(pr, gt)
        loss2 = self.binary_focal(1-pr, 1-gt)
        loss = loss1 + loss2
        return loss.mean()

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    @staticmethod
    def binary_cross_entropy(pr, gt, eps=1e-6):#alpha=0.25
        pr = torch.clamp(pr, eps, 1-eps)
        loss1 = -gt     *torch.log(pr) 
        loss2 = -(1-gt) *torch.log((1-pr))   
        return loss1, loss2 
        
    def forward(self, pr, gt, eps=1e-6, *args):#alpha=0.25
        loss1, loss2 = self.binary_cross_entropy(pr, gt) 
        return (loss1 + loss2).mean()#.item()

class DiceLoss(nn.Module):
    __name__ = 'DiceLoss'
    # DSC(A, B) = 2 * |A ^ B | / ( | A|+|B|)
    def __init__(self, ):
        super(DiceLoss, self).__init__()
        self.func = self.dice
    def forward(self, pr, gt, **args):
        return 2-self.dice(pr,gt)-self.dice(1-pr,1-gt)
        # return 1-self.func(pr, gt)
    @staticmethod
    def dice(pr, gt, smooth=1):#self, 
        pr,gt = pr.view(-1),gt.view(-1)
        inter = (pr*gt).sum()
        union = (pr+gt).sum()
        return (smooth + 2*inter) / (smooth + union)#*0.1

class FusionLoss(nn.Module):
    def __init__(self, *args):
        super(FusionLoss, self).__init__()
        self.losses = nn.ModuleList([*args])
    def forward(self, pr, gt):
        return sum([m(pr, gt) for m in self.losses])

def get_loss(mode='fr'):
    print('loss:', mode)
    if mode=='fr':
        return FocalLoss()
    elif mode=='ce':
        return BCELoss()
    elif mode=='di':
        return DiceLoss()
    elif mode=='l2':
        return nn.MSELoss(reduction='mean')
        
    elif mode=='fd':
        return FusionLoss(FocalLoss(), DiceLoss())
    elif mode=='cd':
        return FusionLoss(BCELoss(), DiceLoss())
    elif mode=='1d':
        return FusionLoss(nn.L1Loss(reduction='mean'), DiceLoss())
    elif mode=='2d':
        return FusionLoss(nn.MSELoss(reduction='mean'), DiceLoss())
        
    elif mode=='frp':
        return FusionLoss(FocalLoss(), PerceptionLoss())
    elif mode=='frt':
        return FusionLoss(FocalLoss(), TextureLoss())
    else:
        raise NotImplementedError()

def swish(x):
	# return x * torch.sigmoid(x)   #计算复杂
	return x * F.relu6(x+3)/6       #计算简单

class FReLU(nn.Module):
	r""" FReLU formulation, with a window size of kxk. (k=3 by default)"""
	def __init__(self, in_channels, *args):
		super().__init__()
		self.func = nn.Sequential(
			nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
			nn.BatchNorm2d(in_channels)
		)
	def forward(self, x):
		x = torch.max(x, self.func(x))
		return x

class DisOut(nn.Module):
	def __init__(self, drop_prob=0.5, block_size=6, alpha=1.0):
		super(DisOut, self).__init__()

		self.drop_prob = drop_prob      
		self.weight_behind=None
  
		self.alpha=alpha
		self.block_size = block_size
		
	def forward(self, x):
		if not self.training: 
			return x

		x=x.clone()
		if x.dim()==4:           
			width=x.size(2)
			height=x.size(3)

			seed_drop_rate = self.drop_prob* (width*height) / self.block_size**2 / (( width -self.block_size + 1)*( height -self.block_size + 1))
			
			valid_block_center=torch.zeros(width,height,device=x.device).float()
			valid_block_center[int(self.block_size // 2):(width - (self.block_size - 1) // 2),int(self.block_size // 2):(height - (self.block_size - 1) // 2)]=1.0
			valid_block_center=valid_block_center.unsqueeze(0).unsqueeze(0)
			
			randdist = torch.rand(x.shape,device=x.device)
			block_pattern = ((1 -valid_block_center + float(1 - seed_drop_rate) + randdist) >= 1).float()
		
			if self.block_size == width and self.block_size == height:            
				block_pattern = torch.min(block_pattern.view(x.size(0),x.size(1),x.size(2)*x.size(3)),dim=2)[0].unsqueeze(-1).unsqueeze(-1)
			else:
				block_pattern = -F.max_pool2d(input=-block_pattern, kernel_size=(self.block_size, self.block_size), stride=(1, 1), padding=self.block_size // 2)

			if self.block_size % 2 == 0:
					block_pattern = block_pattern[:, :, :-1, :-1]
			percent_ones = block_pattern.sum() / float(block_pattern.numel())

			if not (self.weight_behind is None) and not(len(self.weight_behind)==0):
				wtsize=self.weight_behind.size(3)
				weight_max=self.weight_behind.max(dim=0,keepdim=True)[0]
				sig=torch.ones(weight_max.size(),device=weight_max.device)
				sig[torch.rand(weight_max.size(),device=sig.device)<0.5]=-1
				weight_max=weight_max*sig 
				weight_mean=weight_max.mean(dim=(2,3),keepdim=True)
				if wtsize==1:
					weight_mean=0.1*weight_mean
				#print(weight_mean)
			mean=torch.mean(x).clone().detach()
			var=torch.var(x).clone().detach()

			if not (self.weight_behind is None) and not(len(self.weight_behind)==0):
				dist=self.alpha*weight_mean*(var**0.5)*torch.randn(*x.shape,device=x.device)
			else:
				dist=self.alpha*0.01*(var**0.5)*torch.randn(*x.shape,device=x.device)

		x=x*block_pattern
		dist=dist*(1-block_pattern)
		x=x+dist
		x=x/percent_ones
		return x

class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)

class ParallelPolarizedSelfAttention(nn.Module):
    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax=nn.Softmax(1)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        b, c, h, w = x.size()

        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x) #bs,c//2,h,w
        channel_wq=self.ch_wq(x) #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x

        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(x) #bs,c//2,h,w
        spatial_wq=self.sp_wq(x) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*x
        out=spatial_out+channel_out
        return out
class SequentialPolarizedSelfAttention(nn.Module):
    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax=nn.Softmax(1)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        b, c, h, w = x.size()

        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x) #bs,c//2,h,w
        channel_wq=self.ch_wq(x) #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x

        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(channel_out) #bs,c//2,h,w
        spatial_wq=self.sp_wq(channel_out) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*channel_out
        out=spatial_out+channel_out
        return out

class BasicConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, 
				stride=1, padding=1, dilation=1, groups=1, bias=False, bn=True, 
				out='dis', activation=swish, conv=nn.Conv2d, 
				):#'frelu',nn.ReLU(inplace=False),sinlu
		super(BasicConv2d, self).__init__()
		if not isinstance(kernel_size, tuple):
			if dilation>1:
				padding = dilation*(kernel_size//2)	#AtrousConv2d
			elif kernel_size==stride:
				padding=0
			else:
				padding = kernel_size//2			#BasicConv2d

		self.c = conv(in_channels, out_channels, 
			kernel_size=kernel_size, stride=stride, 
			padding=padding, dilation=dilation, bias=bias)
		self.b = nn.BatchNorm2d(out_channels, momentum=0.01) if bn else nn.Identity()

		self.o = nn.Identity()
		drop_prob=0.15
		# self.o = DisOut(drop_prob=drop_prob)#
		self.o = nn.Dropout2d(p=drop_prob, inplace=False) 

		if activation=='frelu':
			self.a = FReLU(out_channels)
		elif activation is None:
			self.a = nn.Identity()
		else:
			self.a = activation

	def forward(self, x):
		x = self.c(x)# x = torch.clamp_max(x, max=99)
		# print('c:', x.max().item())
		x = self.o(x)
		# print('o:', x.max().item())
		x = self.b(x)
		# print('b:', x.max().item())
		x = self.a(x)
		# print('a:', x.max().item())
		return x

class DemoConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, 
				stride=1, padding=1, dilation=1, groups=1, bias=False, bn=True, 
				out='dis', activation=swish, conv=nn.Conv2d, 
				):#'frelu',nn.ReLU(inplace=False),sinlu
		super(DemoConv2d, self).__init__()
		if not isinstance(kernel_size, tuple):
			if dilation>1:
				padding = dilation*(kernel_size//2)	#AtrousConv2d
			elif kernel_size==stride:
				padding=0
			else:
				padding = kernel_size//2			#BasicConv2d

		self.c = conv(in_channels, out_channels, 
			kernel_size=kernel_size, stride=stride, 
			padding=padding, dilation=dilation, bias=bias)
		self.b = nn.BatchNorm2d(out_channels, momentum=0.01) if bn else nn.Identity()
		self.a = nn.ReLU()

	def forward(self, x):
		x = self.c(x)# x = torch.clamp_max(x, max=99)
		# print('c:', x.max().item())
		x = self.b(x)
		# print('b:', x.max().item())
		x = self.a(x)
		# print('a:', x.max().item())
		return x

class PyridConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, 
				stride=1, padding=1, dilation=1, groups=1, bias=False, bn=True, 
				):#ACTIVATION,nn.PReLU(),torch.sin
		super(PyridConv2d, self).__init__()
		self.c = BasicConv2d(in_channels=in_channels, out_channels=out_channels, 
				stride=stride, bn=True, conv=PyConv3, groups=in_channels)
	def forward(self, x):
		return self.c(x)

class CDiff(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, 
				stride=1, padding=1, dilation=1, groups=1, bias=False, bn=True, 
				):#ACTIVATION,nn.PReLU(),torch.sin
		super(CDiff, self).__init__()
		self.c = BasicConv2d(in_channels=in_channels, out_channels=out_channels, 
				stride=stride, bn=True, conv=CDCConv, groups=in_channels)
	def forward(self, x):
		return self.c(x)

class DoverConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, 
				stride=1, padding=1, dilation=1, groups=1, bias=False, bn=True, 
				out='dis', activation=F.gelu,
				):#ACTIVATION,nn.PReLU(),torch.sin
		super(DoverConv2d, self).__init__()
		self.c = BasicConv2d(in_channels=in_channels, out_channels=out_channels, 
				stride=stride, bn=True, conv=DoConv)
	def forward(self, x):
		return self.c(x)

class Bottleneck(nn.Module):
	MyConv = BasicConv2d
	def __init__(self, in_c, out_c, stride=1, downsample=None, **args):
		super(Bottleneck, self).__init__()

		self.conv1 = self.MyConv(in_c, out_c, kernel_size=3, stride=stride)
		self.conv2 = nn.Sequential(
			nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
			# DisOut(),#prob=0.2
			nn.BatchNorm2d(out_c)
		)
		self.relu = swish#nn.LeakyReLU()
		if downsample is None and in_c != out_c:
			downsample = nn.Sequential(
				nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride),
				# DisOut(),#prob=0.2
				nn.BatchNorm2d(out_c),
			)
		self.downsample = downsample

	def forward(self, x):
		residual = x
		if self.downsample is not None:
			residual = self.downsample(x)
		# print('Basic:', x.shape)
		out = self.conv1(x)
		out = self.conv2(out)
		out = self.relu(out + residual)
		# print(out.min().item(), out.max().item())
		return out

class ConvBlock(torch.nn.Module):
	attention=None
	MyConv = BasicConv2d
	def __init__(self, inp_c, out_c, ksize=3, shortcut=False, pool=True):
		super(ConvBlock, self).__init__()
		self.shortcut = nn.Sequential(nn.Conv2d(inp_c, out_c, kernel_size=1), nn.BatchNorm2d(out_c))
		pad = (ksize - 1) // 2

		if pool: self.pool = nn.MaxPool2d(kernel_size=2)
		else: self.pool = False

		block = []
		block.append(self.MyConv(inp_c, out_c, kernel_size=ksize, padding=pad))

		if self.attention=='ppolar':
			# print('ppolar')
			block.append(ParallelPolarizedSelfAttention(out_c))
		elif self.attention=='spolar':
			# print('spolar')
			block.append(SequentialPolarizedSelfAttention(out_c))
		elif self.attention=='siamam':
			# print('siamam')
			block.append(simam_module(out_c))
		# else:
		# 	print(self.attention)
		block.append(self.MyConv(out_c, out_c, kernel_size=ksize, padding=pad))
		self.block = nn.Sequential(*block)
	def forward(self, x):
		if self.pool: x = self.pool(x)
		out = self.block(x)
		return swish(out + self.shortcut(x))

# 输出层 & 下采样
class OutSigmoid(nn.Module):
	def __init__(self, inp_planes, out_planes=1, out_c=8):
		super(OutSigmoid, self).__init__()
		self.cls = nn.Sequential(
			nn.Conv2d(in_channels=inp_planes, out_channels=out_c, kernel_size=3, padding=1, bias=False),
			# nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_c),
			nn.Conv2d(in_channels=out_c, out_channels=1, kernel_size=1, bias=True),
			nn.Sigmoid()
		)
	def forward(self, x):
		return self.cls(x)

class UpsampleBlock(torch.nn.Module):
	def __init__(self, inp_c, out_c, up_mode='transp_conv'):
		super(UpsampleBlock, self).__init__()
		block = []
		if up_mode == 'transp_conv':
			block.append(nn.ConvTranspose2d(inp_c, out_c, kernel_size=2, stride=2))
		elif up_mode == 'up_conv':
			block.append(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False))
			block.append(nn.Conv2d(inp_c, out_c, kernel_size=1))
		else:
			raise Exception('Upsampling mode not supported')

		self.block = nn.Sequential(*block)

	def forward(self, x):
		out = self.block(x)
		return out

class ConvBridgeBlock(torch.nn.Module):
	def __init__(self, out_c, ksize=3):
		super(ConvBridgeBlock, self).__init__()
		pad = (ksize - 1) // 2
		block=[]

		block.append(nn.Conv2d(out_c, out_c, kernel_size=ksize, padding=pad))
		block.append(nn.LeakyReLU())
		block.append(nn.BatchNorm2d(out_c))

		self.block = nn.Sequential(*block)

	def forward(self, x):
		out = self.block(x)
		return out

class UpConvBlock(torch.nn.Module):
	def __init__(self, inp_c, out_c, ksize=3, up_mode='up_conv', conv_bridge=False, shortcut=False):
		super(UpConvBlock, self).__init__()
		self.conv_bridge = conv_bridge

		self.up_layer = UpsampleBlock(inp_c, out_c, up_mode=up_mode)
		self.conv_layer = ConvBlock(2 * out_c, out_c, ksize=ksize, shortcut=shortcut, pool=False)
		if self.conv_bridge:
			self.conv_bridge_layer = ConvBridgeBlock(out_c, ksize=ksize)

	def forward(self, x, skip):
		up = self.up_layer(x)
		if self.conv_bridge:
			out = torch.cat([up, self.conv_bridge_layer(skip)], dim=1)
		else:
			out = torch.cat([up, skip], dim=1)
		out = self.conv_layer(out)
		return out

class LUNet(nn.Module):
	__name__ = 'lunet'
	use_render = False
	def __init__(self, inp_c=1, n_classes=1, layers=(32,32,32,32,32), num_emb=128):
		super(LUNet, self).__init__()
		self.num_features = layers[-1]

		self.__name__ = 'u{}x{}'.format(len(layers), layers[0])
		self.n_classes = n_classes
		self.first = BasicConv2d(inp_c, layers[0])

		self.down_path = nn.ModuleList()
		for i in range(len(layers) - 1):
			block = ConvBlock(inp_c=layers[i], out_c=layers[i + 1], pool=True)
			self.down_path.append(block)

		self.up_path = nn.ModuleList()
		reversed_layers = list(reversed(layers))
		for i in range(len(layers) - 1):
			block = UpConvBlock(inp_c=reversed_layers[i], out_c=reversed_layers[i + 1])
			self.up_path.append(block)

		self.conv_bn = nn.Sequential(
			nn.Conv2d(layers[0], layers[0], kernel_size=1),
			# nn.Conv2d(n_classes, n_classes, kernel_size=1),
			nn.BatchNorm2d(layers[0]),
		)
		self.aux = nn.Sequential(
			nn.Conv2d(layers[0], n_classes, kernel_size=1),
			# nn.Conv2d(n_classes, n_classes, kernel_size=1),
			nn.BatchNorm2d(n_classes),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.first(x)
		down_activations = []
		for i, down in enumerate(self.down_path):
			down_activations.append(x)
			# print(x.shape)
			x = down(x)
		down_activations.reverse()

		for i, up in enumerate(self.up_path):
			x = up(x, down_activations[i])

		# self.feat = F.normalize(x, dim=1, p=2)
		x = self.conv_bn(x)
		self.feat = x

		self.pred = self.aux(x)
		return self.pred

def bunet(**args):
	ConvBlock.attention = DemoConv2d
	net = LUNet(**args)
	net.__name__ = 'bunet'
	return net

def lunet(**args):
	ConvBlock.attention = None
	net = LUNet(**args)
	net.__name__ = 'lunet'
	return net

def munet(**args):
	ConvBlock.attention = 'siamam'
	net = LUNet(**args)
	net.__name__ = 'munet'
	return net
	
def punet(**args):
	ConvBlock.attention='ppolar'
	net = LUNet(**args)
	net.__name__ = 'punet'
	return net
	
def sbau(**args):
	ConvBlock.attention='spolar'
	ConvBlock.MyConv = BasicConv2d
	net = LUNet(**args)
	net.__name__ = 'sbau'
	return net
def sdou(**args):
	ConvBlock.attention='spolar'
	ConvBlock.MyConv = DoverConv2d
	net = LUNet(**args)
	net.__name__ = 'sdou'
	return net
def spyu(**args):
	ConvBlock.attention='spolar'
	ConvBlock.MyConv = PyridConv2d
	net = LUNet(**args)
	net.__name__ = 'spyu'
	return net
def scdu(**args):
	ConvBlock.attention='spolar'
	ConvBlock.MyConv = CDiff
	net = LUNet(**args)
	net.__name__ = 'scdu'
	return net
def sunet(**args):
	ConvBlock.attention = None
	net = LUNet(layers=(32,16,8,4,1), **args)
	net.__name__ = 'sunet'
	return net

def points_selection_hard(feat, prob, true, card=512, dis=100, **args):#point selection by ranking
	assert len(feat.shape)==2, 'feat should contains N*L two dims!'
	L = feat.shape[-1]
	# print(feat.shape, true.shape)
	feat = feat[true.view(-1,1).repeat(1, L)>.5].view(-1, L)
	with torch.no_grad():
		prob = prob[true>.5].view(-1)
		idx = torch.sort(prob, dim=-1, descending=True)[1]
	# h = torch.index_select(feat, dim=0, index=idx[dis:dis+card])
	# l = torch.index_select(feat, dim=0, index=idx[-dis-card:-dis])
	h = torch.index_select(feat, dim=0, index=idx[:card])
	l = torch.index_select(feat, dim=0, index=idx[-card:])
	# print('lh:', l.shape, h.shape)
	# print(prob[idx[:card]].view(-1)[:9])
	# print(prob[idx[-card:]].view(-1)[:9])
	return h, l

def points_selection_half(feat, prob, true, card=512, **args):#point selection by ranking
	assert len(feat.shape)==2, 'feat should contains N*L two dims!'
	L = feat.shape[-1]
	# print(feat.shape, true.shape)
	feat = feat[true.view(-1,1).repeat(1, L)>.5].view(-1, L)
	with torch.no_grad():
		prob = prob[true>.5].view(-1)
		idx = torch.sort(prob, dim=-1, descending=False)[1]
		idx_l, idx_h = torch.chunk(idx, chunks=2, dim=0)

		sample1 = idx_h[torch.randperm(idx_h.shape[0])[:card]]
		sample2 = idx_l[torch.randperm(idx_l.shape[0])[:card]]
	# print(prob[sample][:9])
	h = torch.index_select(feat, dim=0, index=sample1)
	# print(prob[sample][:9])
	l = torch.index_select(feat, dim=0, index=sample2)
	# print('lh:', l.shape, h.shape)
	return h, l

class MLPSampler:
	func = points_selection_half
	def __init__(self, mode='hard', top=4, low=1, dis=0, num=512, select3=False, roma=False):
		self.top = top
		self.low = low
		self.dis = dis
		self.num = num
		self.roma = roma
		self.select = self.select3 if select3 else self.select2

		self.func = eval('points_selection_'+mode)

	@staticmethod
	def rand(*args):
		return MLPSampler(mode='rand', num=512).select(*args)
	@staticmethod
	def half(*args):
		return MLPSampler(mode='half', num=512).select(*args)
	def norm(self, *args, roma=False):
		# if self.roma or roma:# random mapping
		# 	dim = args[0].shape[-1]
		# 	rand = torch.randn(dim, dim, device=args[0].device)
		# 	args = [F.normalize(arg @ rand, dim=-1) for arg in args]
		args = [F.normalize(arg, dim=-1) for arg in args]
		if len(args)==1:
			return args[0]
		return args

	def select(self, feat, pred, true, mask=None, ksize=5):
		# assert mode in ['hard', 'semi', 'hazy', 'edge'], 'sample selection mode wrong!'
		# print(feat.shape, true.shape)
		assert feat.shape[-2:]==true.shape[-2:], 'shape of feat & true donot match!'
		assert feat.shape[-2:]==pred.shape[-2:], 'shape of feat & pred donot match!'
		# reshape embeddings
		feat = feat.clone().permute(0,2,3,1).reshape(-1, feat.shape[1])
		true = true.round()
		fh, fl = self.func(feat,   pred, true, top=self.top, low=self.low, dis=self.dis, card=self.num)
		return torch.cat([fh, fl], dim=0)

	def select2(self, feat, pred, true, mask=None, ksize=5):
		# assert mode in ['hard', 'semi', 'hazy', 'edge'], 'sample selection mode wrong!'
		assert feat.shape[-2:]==true.shape[-2:], 'shape of feat & true donot match!'
		assert feat.shape[-2:]==pred.shape[-2:], 'shape of feat & pred donot match!'
		# reshape embeddings
		feat = feat.permute(0,2,3,1).reshape(-1, feat.shape[1])
		# feat = F.normalize(feat, p=2, dim=-1)
		true = true.round()
		back = (F.max_pool2d(true, (ksize, ksize), 1, ksize//2) - true).round()
		# back = (1-true).round()*mask.round()

		fh, fl = self.func(feat,   pred, true, top=self.top, low=self.low, dis=self.dis, card=self.num)
		bh, bl = self.func(feat, 1-pred, back, top=self.top, low=self.low, dis=self.dis, card=self.num)
		# print('mlp_sample:', fh.shape, fl.shape, bh.shape, bl.shape)
		# return [fh, fl, bh, bl]
		# print(mode)
		return torch.cat([fh, fl, bh, bl], dim=0)

	def select3(self, feat, pred, true, mask=None, ksize=5):
		# assert mode in ['hard', 'semi', 'hazy', 'edge'], 'sample selection mode wrong!'
		# print(feat.shape, true.shape)
		assert feat.shape[-2:]==true.shape[-2:], 'shape of feat & true donot match!'
		assert feat.shape[-2:]==pred.shape[-2:], 'shape of feat & pred donot match!'
		# reshape embeddings
		feat = feat.clone().permute(0,2,3,1).reshape(-1, feat.shape[1])
		# feat = F.interpolate(feat, size=true.shape[-2:], mode='bilinear', align_corners=True)
		# feat = F.normalize(feat, p=2, dim=-1)
		true = true.round()
		dilate = F.max_pool2d(true, (ksize, ksize), stride=1, padding=ksize//2).round()
		edge = (dilate - true).round()
		back = (1-dilate).round()*mask.round()

		# plt.subplot(131),plt.imshow(true.squeeze().data.numpy())
		# plt.subplot(132),plt.imshow(edge.squeeze().data.numpy())
		# plt.subplot(133),plt.imshow(back.squeeze().data.numpy())
		# plt.show()

		# assert back.sum()>0, 'back has no pixels!'
		# assert true.sum()>0, 'true has no pixels!'
		# print('true:', true.sum().item(), true.sum().item()/true.numel())
		# print('edge:', edge.sum().item(), edge.sum().item()/edge.numel())
		# print('back:', back.sum().item(), back.sum().item()/back.numel())
		# print(feat.shape, pred.shape, true.shape)

		fh, fl = self.func(feat,   pred, true, top=self.top, low=self.low, dis=self.dis, card=self.num*2)
		eh, el = self.func(feat, 1-pred, edge, top=self.top, low=self.low, dis=self.dis, card=self.num)
		bh, bl = self.func(feat, 1-pred, back, top=self.top, low=self.low, dis=self.dis, card=self.num)
		# print('mlp_sample:', fh.shape, fl.shape, bh.shape, bl.shape)
		# return [fh, fl, bh, bl]
		# print(mode)
		return torch.cat([fh, fl, eh, el, bh, bl], dim=0)
		# return torch.cat([fh, fl, eh, bh, el, bl], dim=0)

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
	tsne = TSNE(n_components=2, random_state=2021, init='pca')
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

def similar_matrix2(q, k, temperature=0.1):#负太多了
	# print('similar_matrix2:', q.shape, k.shape)
	qfh, qfl, qbh, qbl = torch.chunk(q, 4, dim=0)
	kfh, kfl, kbh, kbl = torch.chunk(k, 4, dim=0)

	# negative logits: NxK
	l_pos = torch.einsum('nc,kc->nk', [qfl, kfh])
	l_neg = torch.einsum('nc,kc->nk', [qbl, kbh])
	# print(l_pos.shape, l_neg.shape)
	return 2 - l_pos.mean() - l_neg.mean()

CLLOSSES = {
	'sim2':similar_matrix2, 'sim3':similar_matrix2,
	}

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
		if clloss in CLLOSSES:
			self.loss = CLLOSSES[clloss]
		else:
			self.regular = self.sphere
			self.loss = get_sphere(clloss, proj_num_length)
		self.encoder = encoder
		# self.__name__ = self.encoder.__name__
		self.__name__ = 'X'.join([self.__name__, self.encoder.__name__]) #, clloss
		
		self.temperature = temperature
		self.proj_num_layers = proj_num_layers
		self.pred_num_layers = pred_num_layers

		self.projector = self.encoder.projector
		self.predictor = self.encoder.predictor

	def forward(self, img, **args):#只需保证两组数据提取出的特征标签顺序一致即可
		out = self.encoder(img, **args)
		self.pred = self.encoder.pred
		self.feat = self.encoder.feat
		# self._dequeue_and_enqueue(proj1_ng, proj2_ng)
		if hasattr(self.encoder, 'tmp'):
			self.tmp = self.encoder.tmp
		return out

	def constraint(self, **args):
		return self.encoder.constraint(**args)

	def sphere(self, sampler, lab, fov=None):#contrastive loss split by classification
		feat = sampler.select(self.feat, self.pred.detach(), lab, fov)
		feat = self.projector(feat)
		self.proj = feat
		true = torch.zeros(size=(feat.shape[0],), dtype=torch.long).to(feat.device)
		true[:feat.shape[0]//2] = 1
		# print('regular:', feat.shape, true.shape)
		return self.loss(feat, true)

	def regular(self, sampler, lab, fov=None):#contrastive loss split by classification
		feat = sampler.select(self.feat.clone(), self.pred.detach(), lab, fov)
		# print(emb.shape)
		proj = self.projector(feat)
		self.proj = proj
		pred = self.predictor(proj)
		# random mapping
		# pred, proj = sampler.norm(pred, proj)
		# rand = torch.randn(64, 64, device=feat.device)
		# pred = F.normalize(pred @ rand, dim=-1)
		# proj = F.normalize(proj @ rand, dim=-1)

		# compute loss
		losSG1 = self.loss(pred, proj.detach(), temperature=self.temperature)
		losSG2 = self.loss(proj, pred.detach(), temperature=self.temperature)
		return losSG1 + losSG2

class MlpSphere(nn.Module):#球极平面射影
	def __init__(self, dim_inp=64, dim_out=32):
		super(MlpSphere, self).__init__()
		# dim_mid = max(dim_inp, dim_out)//2
		# hidden layers
		linear_hidden = []#nn.Identity()
		# for i in range(num_layers - 1):
		linear_hidden.append(nn.Linear(dim_inp, dim_out))
		linear_hidden.append(nn.Dropout(p=0.2))
		linear_hidden.append(nn.BatchNorm1d(dim_out))
		linear_hidden.append(nn.LeakyReLU())
		self.linear_hidden = nn.Sequential(*linear_hidden)

	def forward(self, x):
		x = self.linear_hidden(x)
		return x

class MlpNorm(nn.Module):
	def __init__(self, dim_inp=256, dim_out=64):
		super(MlpNorm, self).__init__()
		dim_mid = min(dim_inp, dim_out)#max(dim_inp, dim_out)//2
		# hidden layers
		linear_hidden = []#nn.Identity()
		# for i in range(num_layers - 1):
		linear_hidden.append(nn.Linear(dim_inp, dim_mid))
		linear_hidden.append(nn.Dropout(p=0.2))
		linear_hidden.append(nn.BatchNorm1d(dim_mid))
		linear_hidden.append(nn.LeakyReLU())
		self.linear_hidden = nn.Sequential(*linear_hidden)

		self.linear_out = nn.Linear(dim_mid, dim_out)# if num_layers >= 1 else nn.Identity()

	def forward(self, x):
		x = self.linear_hidden(x)
		x = self.linear_out(x)
		return F.normalize(x, p=2, dim=-1)

def torch_dilation(x, ksize=3, stride=1):
	return F.max_pool2d(x, (ksize, ksize), stride, ksize//2)

class MorphBlock(nn.Module):
	def __init__(self, inp_ch=2, channel=8):
		super().__init__()
		self.ch_wv = nn.Sequential(
			nn.Conv2d(inp_ch,channel,kernel_size=5, padding=2),
			nn.Conv2d(channel,channel,kernel_size=5, padding=2),
			nn.BatchNorm2d(channel),
			nn.Conv2d(channel,channel//2,kernel_size=3, padding=1),
		)
		self.ch_wq = nn.Sequential(
			nn.Conv2d(channel//2,8,kernel_size=3, padding=1),
			nn.BatchNorm2d(8),
			nn.Conv2d(8,1,kernel_size=1),
			nn.Sigmoid()
		)
	
	def forward(self, x, o):
		x = torch.cat([torch_dilation(o, ksize=3), x, o], dim=1)#, 1-torch_dilation(1-x, ksize=3)
		x = self.ch_wv(x)
		# print(x.shape)
		return self.ch_wq(x)

class SeqNet(nn.Module):#Supervised contrastive learning segmentation network
	__name__ = 'scls'
	def __init__(self, type_net, type_seg, num_emb=128):
		super(SeqNet, self).__init__()

		self.fcn = eval(type_net+'(num_emb=num_emb)')#build_model(cfg['net']['fcn'])
		self.seg = eval(type_seg+'(inp_c=32)')#build_model(cfg['net']['seg'])

		self.projector = MlpNorm(32, num_emb)#self.fcn.projector#MlpNorm(32, 64, num_emb)
		self.predictor = MlpNorm(num_emb, num_emb)#self.fcn.predictor#MlpNorm(32, 64, num_emb)

		self.morpholer1 = MorphBlock(32+2)#形态学模块使用一个还是两个哪？
		self.morpholer2 = MorphBlock(32+2)#形态学模块使用一个还是两个哪？
		self.__name__ = '{}X{}'.format(self.fcn.__name__, self.seg.__name__)

	def constraint(self, aux=None, fun=None, **args):
		aux = torch_dilation(aux)
		los1 = fun(self.sdm1, aux)
		los2 = fun(self.sdm2, aux)
		# if self.__name__.__contains__('dmf'):
		# 	los1 = los1 + self.fcn.regular()*0.1
		return los1, los2
	
	def regular(self, sampler, lab, fov=None, return_loss=True):
		emb = sampler.select(self.feat.clone(), self.pred.detach(), lab, fov)
		# print(emb.shape)
		emb = self.projector(emb)
		# print(emb.shape)
		self.emb = emb
		if return_loss:
			return sampler.infonce(emb)
	tmp = {}
	def forward(self, x):
		aux = self.fcn(x)
		self.feat = self.fcn.feat
		out = self.seg(self.feat)
		self.pred = out
		# print(self.fcn.feat.shape, self.seg.feat.shape)
		self.sdm1 = self.morpholer1(self.fcn.feat, aux)
		self.sdm2 = self.morpholer2(self.seg.feat, out)
		self.tmp = {'sdm1':self.sdm1, 'sdm2':self.sdm2}

		if self.training:
			if isinstance(aux, (tuple, list)):
				return [self.pred, aux[0], aux[1]]
			else:
				return [self.pred, aux]
		return self.pred

class SphereUNet(nn.Module):
	__name__ = 'sphu'
	def __init__(self, inp_c=1, n_classes=1, layers=(32,32,32,32,32), num_emb=32):
		super(SphereUNet, self).__init__()
		self.num_features = layers[-1]
		self.n_classes = n_classes
		self.first = BasicConv2d(inp_c, layers[0])

		self.down_path = nn.ModuleList()
		for i in range(len(layers) - 1):
			block = ConvBlock(inp_c=layers[i], out_c=layers[i + 1], pool=True)
			self.down_path.append(block)

		self.up_path = nn.ModuleList()
		reversed_layers = list(reversed(layers))
		for i in range(len(layers) - 1):
			block = UpConvBlock(inp_c=reversed_layers[i], out_c=reversed_layers[i + 1])
			self.up_path.append(block)

		self.projector = MlpNorm(layers[0], num_emb)
		self.spherepro = MlpSphere(num_emb, 32)
		self.out = OutSigmoid(32, 1)
		self.morpholer = MorphBlock(32+2)#形态学模块使用一个还是两个哪？

		self.conv_bn = nn.Sequential(
			nn.Conv2d(layers[0], layers[0], kernel_size=1),
			# nn.Conv2d(n_classes, n_classes, kernel_size=1),
			nn.BatchNorm2d(layers[0]),
		)

	def constraint(self, aux=None, fun=None, **args):
		aux = torch_dilation(aux)
		los = fun(self.sdm, aux)
		return los

	def regular(self, sampler, lab, fov=None, *args):
		feat = sampler.select(self.feat, self.pred.detach(), lab, fov)
		self.feat = feat
		
		glb_sim = - (feat.detach() @ feat.permute(1,0)).mean()
		glb_agn = align_loss(feat, torch.flip(feat, dims=[0,]))
		glb_mse = F.mse_loss(feat, torch.flip(feat, dims=[0,]))
		return glb_sim + glb_agn + glb_mse

	def forward(self, x):
		x = self.first(x)
		down_activations = []
		for i, down in enumerate(self.down_path):
			down_activations.append(x)
			# print(x.shape)
			x = down(x)
		down_activations.reverse()

		for i, up in enumerate(self.up_path):
			x = up(x, down_activations[i])

		# self.feat = F.normalize(x, dim=1, p=2)
		x = self.conv_bn(x)

		B,C,H,W = x.shape
		x = self.projector(x.permute(0,2,3,1).reshape(-1, C))
		self.proj = x.clone()#.reshape(-1, C)
		self.feat = self.proj.reshape(B,-1,self.proj.shape[-1]).permute(0,2,1).reshape(B, -1, H, W)

		out = self.spherepro(self.proj)
		# print('proj & pred:', self.proj.shape, self.pred.shape, out.shape)#torch.Size([2, 128, 64, 64])
		out = out.reshape(B,-1,out.shape[-1]).permute(0,2,1).reshape(B, -1, H, W)
		# print('sphere proj:', out.shape)#torch.Size([2, 128, 64, 64])
		self.pred = self.out(out)

		self.sdm = self.morpholer(out, self.pred)
		self.tmp = {'sdm':self.sdm}
		return self.pred
def spu(**args):
	return SphereUNet(**args)

def build_model(type_net='hrdo', type_seg='', type_loss='sim2', type_arch='', num_emb=128):

	if type_net == 'hrdo':
		model = hrdo()
	elif type_net == 'dou':
		model = DoU()
	elif type_net == 'dmf':
		model = dmf32()
	else:
		model = eval(type_net+'(num_emb=num_emb)')
		# raise NotImplementedError(f'--> Unknown type_net: {type_net}')

	if type_seg!='':
		model = SeqNet(type_net, type_seg, num_emb=num_emb)
		if type_arch=='siam':
			model = SIAM(encoder=model, clloss=type_loss, proj_num_length=num_emb)
			model.__name__ = model.__name__.replace('siamXlunetXlunet', 'SLL')
		elif type_arch=='roma':
			model = ROMA(encoder=model, clloss=type_loss, proj_num_length=num_emb)

	return model

import os, glob, sys, time, torch
from torch.optim import lr_scheduler
# from torch.cuda import amp
torch.set_printoptions(precision=3)

class GradUtil(object):
	def __init__(self, model, loss='ce', lr=0.01, wd=2e-4, root='.'):
		self.path_checkpoint = os.path.join(root, 'super_params.tar')
		if not os.path.exists(root):
			os.makedirs(root)

		self.lossName = loss
		self.criterion = get_loss(loss)
		params = filter(lambda p:p.requires_grad, model.parameters())
		self.optimizer = RAdamW(params=params, lr=lr, weight_decay=2e-4)
		self.scheduler = ReduceLR(name=loss, optimizer=self.optimizer,  
			mode='min', factor=0.7, patience=2, 
			verbose=True, threshold=0.0001, threshold_mode='rel', 
			cooldown=2, min_lr=1e-5, eps=1e-9)
		
	def isLrLowest(self, thresh=1e-5):
		return self.optimizer.param_groups[0]['lr']<thresh

	coff_ds = 0.5
	def calcGradient(self, criterion, outs, true, fov=None):
		lossSum = 0#torch.autograd.Variable(torch.tensor(0, dtype=torch.float32), requires_grad=True)
		if isinstance(outs, (list, tuple)):
			# ratio = 1/(1+len(outs))
			for i in range(len(outs)-1,0,-1):#第一个元素尺寸最大
				# print('输出形状：', outs[i].shape, true.shape)
				# true = torch.nn.functional.interpolate(true, size=outs[i].shape[-2:], mode='nearest')
				loss = criterion(outs[i], true)#, fov
				lossSum = lossSum + loss*self.coff_ds
			outs = outs[0]
		# print(outs.shape, true.shape)
		lossSum = lossSum + criterion(outs, true)#, fov
		return lossSum
		
	def backward_seg(self, pred, true, fov=None, model=None, requires_grad=True, losInit=[]):
		self.optimizer.zero_grad()

		costList = []
		#torch.autograd.Variable(torch.tensor(0, dtype=torch.float32), requires_grad=True)
		los = self.calcGradient(self.criterion, pred, true, fov)
		costList.append(los)
		self.total_loss += los.item()
		del pred, true, los

		if isinstance(losInit, list) and len(losInit)>0:#hasattr(losInit, 'item'):#not isinstance(losInit, int):
			costList.extend(losInit)

		losSum = sum(costList)
		losStr = ','.join(['{:.4f}'.format(los.item()) for los in costList])
		if requires_grad:
			losSum.backward()#梯度归一化
			#梯度裁剪
			# nn.utils.clip_grad_value_(model.parameters(), clip_value=1.1)#clip_value=1.1
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)#（最大范数，L2)
			self.optimizer.step()
		return losSum.item(), losStr

	total_loss = 0
	def update_scheduler(self, i=0):
		logStr = '\r{:03}# '.format(i)
		# losSum = 0
		logStr += '{}={:.4f},'.format(self.lossName, self.total_loss)
		print(logStr, end='')
		# self.callBackEarlyStopping(los=losSum)

		if isinstance(self.scheduler, ReduceLR):
			self.scheduler.step(self.total_loss)
		else:
			self.scheduler.step()
		self.total_loss = 0

import time, tqdm#, socket
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
		print('*'*32,'fitting:'+self.root) 
		# self.desc()
		for i in range(epochs):
			
			# 投影(想象可视化的时候选点策略选hard negatives 还是hazy negatives???)
			if i%20==0 and self.args.sss!='':	#从数据集中取一张图片用来画图
				self.plot_emb(epoch=i)

			# 训练
			lossItem = self.train()
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
			else:
				self.callBackEarlyStopping(lossItem)#eye does not use this line
			# 早停
			if self.stop_training==True:
				print('Stop Training!!!')
				break
		self.desc()
		if self.evalMetric:
			print(self.bests)
		self.logTxt.append(str(self.bests))
		with open(self.root + '/logs.txt', 'w') as f:
			f.write('\n'.join(self.logTxt))
		if hasattr(self.model, 'tmp'):
			tensorboard_logs(self.model.tmp, root=self.root)
	
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

			# print('backward:', out.shape, lab.shape, fov.shape)
			_lossItem, losStr = self.gradUtil.backward_seg(out, lab, fov, self.model, requires_grad=True, losInit=losInit) 
			lossItem += _lossItem
			del out, lab, fov, aux   
			# print('\r{:03}$ los={:.3f}'.format(i, _lossItem), end='')
			tbar.set_description('{:03}$ {:.3f}={}'.format(i, _lossItem, losStr))
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
			print('\r{:03}$ auc={:.4f} & iou={:.4f} & f1s={:.4f} & a={:.4f}'.format(i, auc, iou, f1s, a), end='')
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

			fov = (fov>.5).astype(np.bool) 
			self.score_pred.score(true[fov], pred[fov])
			# self.score_otsu.score(true[fov], pred[fov], otsu=True)

		self.score_pred.end(model_name=name + ('_tta' if tta else ''))
		# self.score_otsu.end(model_name=name+'_otsu')
		print('Mean inference time:', timeSum/(i+1))

import argparse
def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Unsupported value encountered.')

parser = argparse.ArgumentParser(description="Train network")
#	实验参数	C:对比学习、P：先验知识
parser.add_argument('--inc', type=str, default='', help='instruction')#skeleton & dilation
parser.add_argument('--gpu', type=int, default=0, help='cuda number')
parser.add_argument('--los', type=str, default='fr', help='loss function')
parser.add_argument('--net', type=str, default='lunet', help='network')
parser.add_argument('--seg', type=str, default='lunet', help='network')
# parser.add_argument('--patch', type=str2bool, default=True, help='Patch based!')
parser.add_argument('--csm', type=str2bool, default=False, help='Color Space Mixture!')
parser.add_argument('--coff_ds', type=float, default=0.5, help='Cofficient of Deep Supervision!')

#	数据参数
parser.add_argument('--db', type=str, default='stare', help='instruction')
parser.add_argument('--loo', type=int, default=0, help='Leave One Out')
parser.add_argument('--bs', type=int, default=32, help='batch size')
parser.add_argument('--ds', type=int, default=128, help='data size')
parser.add_argument('--pl', type=str2bool, default=False, help='Parallel!')
parser.add_argument('--root', type=str, default='', help='root folder')
#	正则化参数
parser.add_argument('--ct', type=str2bool, default=True, help='Constraint for Network!')
parser.add_argument('--coff_ct', type=float, default=.9, help='Cofficient of Constraint!')
parser.add_argument('--loss_ct', type=str, default='di', help='Loss of Contrastive learning!')

#	对比学习相关参数
parser.add_argument('--arch', type=str, default='siam', help='architechture')
parser.add_argument('--roma', type=str2bool, default=False, help='Random Mapping!')
# parser.add_argument('--rd', type=str2bool, default=False, help='Render for Contrastive Learning!')
parser.add_argument('--coff_cl', type=float, default=.1, help='Cofficient of Contrastive learning!')
parser.add_argument('--temp_cl', type=float, default=.1, help='Temperature of Contrastive learning!')
parser.add_argument('--loss_cl', type=str, default='sim3', help='Loss of Contrastive learning!')#, choices=['', 'au', 'nce', 'sim', 'nce2', 'sim2']

parser.add_argument('--sss', type=str, default='half', choices=['', 'hard', 'half'], help='Sample Selection Strategy!')
parser.add_argument('--top', type=int, default=4, help='sampler top')
parser.add_argument('--low', type=int, default=2, help='sampler low')
parser.add_argument('--dis', type=int, default=4, help='sampler dis')
parser.add_argument('--num', type=int, default=512, help='sampler number')
# parser.add_argument('--se3', type=str2bool, default=False, help='Select 3 or 2!')
# parser.add_argument('--con', type=str, default='', help='infonce version')

args = parser.parse_args()



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"#
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in range(args.gpu, 4)])

# 训练程序########################################################
if __name__ == '__main__':
	args.inc += ('_csm' if args.csm else '')

	dataset = EyeSetGenerator(dbname=args.db, datasize=args.ds, loo=args.loo) 
	# dataset = EyeSetGenerator(dbname=args.db, isBasedPatch=args.patch) 
	dataset.use_csm = args.csm

	net = build_model(args.net, args.seg, args.loss_cl, args.arch)
	if args.db=='stare':
		net.__name__ += 'LOO'+str(args.loo)

	keras = KerasTorch(model=net, args=args) 
	keras.args = args
	keras.isParallel = args.pl
	
	if args.ct or 'dmf' in net.__name__:
		args.ct = True
		net.__name__ += args.loss_ct + str(args.coff_ct)
		keras.loss_ct = get_loss(args.loss_ct)
	else:
		args.ct = False

	net.__name__ += args.inc + 'ds'+str(args.coff_ds) + args.sss
	if args.sss!='':
		args.se3 = args.loss_cl.endswith('3')
		keras.sampler = MLPSampler(mode=args.sss, select3=args.se3, roma=args.roma,
			top=args.top, low=args.low, dis=args.dis, num=args.num)
		if args.sss=='semi':
			net.__name__ += str(args.dis)
		elif args.sss in ['part', 'prob']:
			net.__name__ += str(args.top) + str(args.low)	#Top & low
		elif args.sss in ['thsh', 'drop']:#, 'hard'
			net.__name__ += str(args.dis)
			
		# if args.arch=='':
		# 	net.__name__ += args.con
		# 	keras.sampler.infonce = ConLoss(con=args.con, temp=args.temp_cl)
		net.__name__ += 'Roma' if args.roma else ''
		net.__name__ += args.loss_cl+'S'+str(args.num) + 'C'+str(args.coff_cl) + 'T'+str(args.temp_cl)	#Cofficiency & Temperature

	print('Network Name:', net.__name__)

	keras.compile(dataset, loss=args.los, lr=0.01)  
	keras.gradUtil.coff_ds = args.coff_ds
	if args.root=='':
		# keras.test(testset=dataset, key='los', flagSave=True)  
		# keras.val()
		keras.fit(epochs=169)    

	keras.test(testset=dataset, key='los', flagSave=True, tta=False)  
	for key in keras.paths.keys():
		keras.test(testset=dataset, key=key, flagSave=True, tta=False)

	if hasattr(keras, 'score_pred'):
		keras.score_pred.desc()

