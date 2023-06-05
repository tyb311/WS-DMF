# -*- coding: utf-8 -*-
import sys
sys.path.append('.')
sys.path.append('..')

if __name__ == '__main__':
	from trans import *
	from show import *
	from eyenpy import *
else:
	try:
		from .trans import *
		from .eyenpy import *
	except:
		from data.trans import *
		from data.eyenpy import *


# import imgaug.augmenters as iaa
# def get_seed():
# 	return random.randint(0,1024)

# IAA_NOISE = iaa.OneOf(children=[# Noise
# 		iaa.Add(per_channel=True),
# 		iaa.AddElementwise(),
# 		iaa.Multiply(per_channel=True),
# 		iaa.MultiplyElementwise(per_channel=True),

# 		iaa.AdditiveGaussianNoise(per_channel=True),
# 		iaa.AdditiveLaplaceNoise(per_channel=True),
# 		iaa.AdditivePoissonNoise(per_channel=True),

# 		iaa.SaltAndPepper(per_channel=True),
# 		iaa.ImpulseNoise(seed=get_seed()),
# 	]
# )
# IAA_BLEND = iaa.OneOf(children=[# Noise
# 		# Blend
# 		iaa.BlendAlpha(factor=(0.0, 1.0), foreground=iaa.Add(seed=get_seed()), background=iaa.Multiply(seed=get_seed())),
# 		iaa.BlendAlpha(factor=(0.0, 1.0), foreground=iaa.Add(seed=get_seed()), background=iaa.Add(seed=get_seed())),
# 		iaa.BlendAlpha(factor=(0.0, 1.0), foreground=iaa.Multiply(seed=get_seed()), background=iaa.Multiply(seed=get_seed())),
# 		iaa.BlendAlpha(factor=(0.0, 1.0), foreground=iaa.Multiply(seed=get_seed()), background=iaa.Add(seed=get_seed())),

# 		iaa.BlendAlphaElementwise(factor=(0.0, 1.0), foreground=iaa.Clouds(get_seed()), seed=get_seed(), per_channel=True),
# 		iaa.BlendAlphaElementwise(factor=(0.0, 1.0), foreground=iaa.AddToBrightness(seed=get_seed()), seed=get_seed(), per_channel=True),
# 		iaa.BlendAlphaElementwise(factor=(0.0, 1.0), foreground=iaa.AddToHue(seed=get_seed()), seed=get_seed(), per_channel=True),
# 		iaa.BlendAlphaElementwise(factor=(0.0, 1.0), foreground=iaa.AddToSaturation(seed=get_seed()), seed=get_seed(), per_channel=True),

# 		iaa.BlendAlphaVerticalLinearGradient(iaa.Clouds(seed=get_seed()), max_value=.5, seed=get_seed()),
# 		iaa.BlendAlphaVerticalLinearGradient(iaa.AddToHue(seed=get_seed()), seed=get_seed()),
# 		iaa.BlendAlphaVerticalLinearGradient(iaa.AddToSaturation(seed=get_seed()), seed=get_seed()),
# 		iaa.BlendAlphaVerticalLinearGradient(iaa.AddToBrightness(seed=get_seed()), seed=get_seed()),

# 		iaa.BlendAlphaHorizontalLinearGradient(iaa.Clouds(seed=get_seed()), max_value=.5, seed=get_seed()),
# 		iaa.BlendAlphaHorizontalLinearGradient(iaa.AddToHue(seed=get_seed()), seed=get_seed()),
# 		iaa.BlendAlphaHorizontalLinearGradient(iaa.AddToSaturation(seed=get_seed()), seed=get_seed()),
# 		iaa.BlendAlphaHorizontalLinearGradient(iaa.AddToBrightness(seed=get_seed()), seed=get_seed()),

# 		iaa.BlendAlphaCheckerboard(7, 7, iaa.AddToHue(seed=get_seed()), seed=get_seed()),
# 	]
# )

#start#
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

TRANS_NOISE1 = iaa.Sequential(children=[IAA_NOISE, IAA_BLEND])
TRANS_NOISE2 = iaa.Sequential(children=[IAA_BLEND, IAA_NOISE])

from albumentations import (
	# 空间
	RGBShift, ChannelDropout, ChannelShuffle, 
	# 色调
	HueSaturationValue, RandomContrast, RandomBrightness, RandomCrop, CenterCrop,
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
import itertools
from torch.utils.data import DataLoader, Dataset
class EyeSetGenerator(Dataset, EyeSetResource):
	exeNums = {'train':Aug4CSA.number, 'val':Aug4CSA.number, 'test':1}#Aug4CSA.number
	exeMode = 'train'#train, val, test
	exeData = 'train'#train, test, full

	expCross = False  
	SIZE_IMAGE = 128# 
	# SIZE_PATCH = itertools.cycle([(384,1), (256,2), (128,8), (64,32)])#
	def __init__(self, **args):
		super(EyeSetGenerator, self).__init__(**args)
		print('EyeSetGernerator')
		# hp,wp = self.size['pad']
		# hc,wc = self.size['raw']
		# self.tran_pade = PadIfNeeded(p=1, min_height=hp, min_width=wp)
		# self.tran_crop = CenterCrop(height=hc, width=wc)
		print('#'*32, 'Patch-Based' if self.isBasedPatch else 'Image-Based')
		
		if self.isBasedPatch:
			self.exeNums = {'train':1, 'val':Aug4CSA.number, 'test':1}#Aug4CSA.number
		else:
			self.exeNums = {'train':Aug4CSA.number, 'val':Aug4CSA.number, 'test':1}#Aug4CSA.number
		if self.dbname=='uoadr':
			self.exeNums['val'] = 1
			self.numXLen = 1
		print('ExeNum:', self.exeNums)
		self.augCrop = CropNonEmptyMaskIfExists(p=1, height=self.SIZE_IMAGE, width=self.SIZE_IMAGE)
	
	numXLen=6
	def __len__(self):
		length = self.lens[self.exeData]*self.exeNums[self.exeMode]
		if self.isTrainMode:
			if self.isBasedPatch:
				return length*3
			return length*self.numXLen#兴许是训练轮数不够多的缘故，也未可知
		return length

	def set_mode(self, mode='train'):
		self.exeMode = mode
		self.exeData = 'full' if self.expCross else mode 
		self.isTrainMode = (mode=='train')
		self.isValMode = (mode=='val')
		self.isTestMode = (mode=='test')
	# hard_samples = []
	# def add2hard_samples(self, idx:int, num:int):## hard sample selection when it 's evaluate mode
	# 	for i in range(num):
	# 		self.hard_samples.append(self.labs['eval'][idx%self.LEN_EVAL])
	def trainSet(self, bs=32):#pin_memory=True, , shuffle=True
		self.set_mode(mode='train')
		# self.labs['train'] = self.hard_samples.copy() if self.hard_samples.__len__()>0 else self.labs['eval']
		# self.hard_samples.clear()	#清空困难样本列表
		# random.shuffle(self.labs['train'])
		# self.LEN_TRAIN = len(self.labs['train'])
		# self.len_mod = len(self.labs['train'])
		return DataLoader(self, batch_size=bs, pin_memory=True, num_workers=4, shuffle=True)
	def valSet(self, bs=1):
		self.set_mode(mode='val')
		return DataLoader(self, batch_size=bs,  pin_memory=True, num_workers=2)
	def testSet(self):
		self.set_mode(mode='test')
		return DataLoader(self, batch_size=1,  pin_memory=True, num_workers=2)
	#DataLoader worker (pid(s) 5220) exited unexpectedly, 令numworkers>1就可以啦
	
	def parse(self, pics, mix=False, csa=False):
		def reshape(x):
			if len(x.shape)==3:
				x = x.unsqueeze(0)
			return x
		# rows, cols = pics['lab'].squeeze().shape[-2:]     
		# pics['img'][:,0] = (pics['img'][:,0]-0.514560938812792) / 0.10452990154735745
		# pics['img'][:,1] = (pics['img'][:,1]-0.602489422261715) / 0.10357838263735175
		# pics['img'] = torch.cat([pics['img'], pics['fov']], dim=1)
		return reshape(pics['img']), reshape(pics['lab']), reshape(pics['fov']), reshape(pics['aux'])
		# return pics['img'], pics['lab'], pics['fov'], pics['aux']

	def post(self, img, lab, fov, **args):#浮点数 & Torch张量都可以输入
		if type(img) is not np.ndarray:img = img.squeeze().cpu().numpy()
		if type(lab) is not np.ndarray:lab = lab.squeeze().cpu().numpy()
		if type(fov) is not np.ndarray:fov = fov.squeeze().cpu().numpy()
		img = self.tran_crop(image=img)['image']
		lab = self.tran_crop(image=lab)['image']
		fov = self.tran_crop(image=fov)['image']
		# img[fov<.5] = 0
		return img, lab, fov
	
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
	def __getitem__(self, idx):
		# print(self.exeData, self.exeMode)
		pics = self.readDict(idx % self.lens[self.exeData], self.exeData)
		# pics.pop('mat')#= (255-pics['lab'])//255*pics['img']

		# pics['aux'] = cv2.cvtColor(pics['img'], cv2.COLOR_RGB2GRAY)
		skel = morphology.skeletonize((pics['lab']/255.0).round()).astype(np.uint8)
		pics['aux'] = morphology.dilation(skel, self.kernel)*255#pics['lab']#

		if self.isTrainMode:
			mask = np.stack([pics['lab'], pics['fov'], pics['aux']], axis=-1)
			if not self.isBasedPatch:# 裁剪增强	Patch-Based
				picaug = self.augCrop(image=pics['img'], mask=mask)
				pics['img'], mask = picaug['image'], picaug['mask']
				
			pics['img'] = TRANS_TEST(image=pics['img'])['image']

			# # 添加噪声
			# pics['img'] = TRANS_NOISE1(image=pics['img'])
			if torch.rand(1).item()>0.5:#random.choice([True, False]):
				pics['img'] = TRANS_NOISE1(image=pics['img'])
			else:
				pics['img'] = TRANS_NOISE2(image=pics['img'])

			# # print(pics['lab'].shape, pics['fov'].shape, pics['aux'].shape)
			picaug = TRANS_AAUG(image=pics['img'], mask=mask)
			pics['img'], mask = picaug['image'], picaug['mask']
			pics['lab'], pics['fov'], pics['aux'] = mask[:,:,0],mask[:,:,1],mask[:,:,2]
			pics = Aug4CSA.forward_train(pics)
		else:
			# 图像补齐
			for key in pics.keys():
				pics[key] = self.tran_pade(image=pics[key])['image']
			pics['img'] = TRANS_TEST(image=pics['img'])['image']

			if self.isValMode:
				flag = idx//self.lens[self.exeData]
				pics = Aug4CSA.forward_val(pics, flag)#怪不得过拟合，验证集重复了样本来着
			else:
				# pics = Aug4CSA.forward_test(pics)
				# pics['img'] = pics['csa']
				pics['img'] = cv2.cvtColor(pics['img'], cv2.COLOR_RGB2GRAY)
				# pics['img'] = pics['img'][:,:,1]#green or gray

		for key in pics.keys():
			# print(key, pics[key].shape)
			pics[key] = torch.from_numpy(pics[key]).type(torch.float32).div(255.0)
			pics[key] = pics[key].view(-1, pics['lab'].shape[-2], pics['lab'].shape[-1]) 
		pics['aux'] = pics['aux'].round()
		pics['lab'] = pics['lab'].round()
		pics['fov'] = pics['fov'].round()
		# pics['img'] = torch.where(pics['fov']>0, (pics['img'] - 0.4) / 0.1, pics['img'])
		# pics['img'] = (pics['img'] - 0.4633) / 0.0812#zero-center is not needed
		return pics
#end#

	# 0.4633122134953737
	# 0.08158384058624506

# 下一步，数据增强要跟上，用嘴头疼的IAA
# 把数据训练集分布打乱，等等，
# 给模型初始部分加点东西，防止分布变动对比只用匹配滤波的效果
import kornia

def main():
	# db = EyeSetGenerator(folder='../datasets/seteye', dbname='drive', isBasedPatch=True)#
	# db = EyeSetGenerator(folder=r'G:\Objects\datasets\seteye', dbname='drive', isBasedPatch=False)#
	# db = EyeSetGenerator(folder=r'G:\Objects\datasets\seteye', dbname='rcslo', isBasedPatch=False)#
	# db = EyeSetGenerator(folder=r'G:\Objects\datasets\seteye', dbname='iostar', isBasedPatch=False)#
	db = EyeSetGenerator(folder=r'G:\Objects\datasets\seteye', dbname='uoadr', isBasedPatch=False)#
	# db.expCross = True
	print('generator:', len(db.trainSet()), len(db.valSet()), len(db.testSet()), )

	# db.expCross=True
	# for i, imgs in enumerate(db.trainSet(8)):
	# for i, imgs in enumerate(db.valSet(1)):
	for i, imgs in enumerate(db.testSet()):
		# print(imgs.keys())
		# print(imgs)
		(img, lab, fov, aux) = db.parse(imgs, csa=True)
		# print(img.dtype, lab.dtype)

		# gauss = kornia.filters.GaussianBlur2d((23, 23), (15, 15))
		# gauss = kornia.filters.MedianBlur((13, 13))

		# roi = img[fov>0.5]
		# aux = (img - roi.mean())/roi.std()
		# aux = torch.clamp(aux+1.5, 0, 3)/3

		# # 显示
		desc(img, shape=True, value=True)
		# desc(fov, shape=True, value=True)
		# desc(lab, shape=True, value=True)
		# desc(aux, shape=True, value=True)

		# show(img)
		# show2(img[0,0:1],img[0,1:2])
		# aux = kornia.filters.gaussian_blur2d(lab, kernel_size=(5,5), sigma=(10,10))
		# show2(aux,lab)
		# show2(img,lab)

		# desc(aux, shape=True, value=True)

		# show4(img, lab, fov, aux)
		# img, lab, fov = db.post(img, lab, fov, 3)
		# show3(img, lab, fov)
		show4(img, lab, fov, aux)
		# show4(img[0,0:1], lab, fov, aux)

		if i>5:
			break

#	可以试想先把lab膨胀，进行训练，结束后做腐蚀处理来保证细血管的识别
#	是否在线选择困难样本？
#	是否有必要使用离线裁剪样本？
#	发现噪声加的不够，没有原来噪声强，怪不得过拟合
if __name__ == '__main__':
	main()

	
	# db = EyeSetGenerator(folder='../datasets/seteye', dbname='drive', isBasedPatch=True)#
	# # db = EyeSetGenerator(folder=r'G:\Objects\expSeg\datasets\seteye', dbname='drive', isBasedPatch=False)#
	# # db.expCross = True
	# print('generator:', len(db.trainSet()), len(db.valSet()), len(db.testSet()), )
	
	# db.testSet()
	# imgs = db.__getitem__(0)
	# (img, lab, fov, aux) = db.parse(imgs, csa=True)
	# # print(img.dtype, lab.dtype)

	# # gauss = kornia.filters.GaussianBlur2d((23, 23), (15, 15))
	# # gauss = kornia.filters.MedianBlur((13, 13))

	# # roi = img[fov>0.5]
	# # aux = (img - roi.mean())/roi.std()
	# # aux = torch.clamp(aux+1.5, 0, 3)/3

	# # # 显示
	# desc(img, shape=True, value=True)
	# show3(img, lab, fov)