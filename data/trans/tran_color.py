if __name__ == '__main__':
	from tran import *
else:
	from .tran import *



# class ChannelGYH(object):
#     def __call__(self, pic):
#         g =  cv2.split(pic['img'])[1]
#         y =  cv2.split(cv2.cvtColor(pic['img'], cv2.COLOR_RGB2YCrCb))[0]
#         h =  cv2.split(cv2.cvtColor(pic['img'], cv2.COLOR_RGB2LAB))[0]
#         pic['img'] = np.stack([g,y,h], axis=-1)
#         return pic

#start#
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
	if tran==cv2.COLOR_RGB2LAB:
		rgb = cv2.split(rgb)[0]
	elif tran==cv2.COLOR_RGB2XYZ:
		rgb = cv2.split(rgb)[0]
	elif tran==cv2.COLOR_RGB2LUV:
		rgb = cv2.split(rgb)[0]
	elif tran==cv2.COLOR_RGB2HLS:
		rgb = cv2.split(rgb)[1]
	elif tran==cv2.COLOR_RGB2YCrCb:
		rgb = cv2.split(rgb)[0]
	elif tran==cv2.COLOR_RGB2YUV:
		rgb = cv2.split(rgb)[0]
	elif tran==cv2.COLOR_RGB2BGR:
		rgb = cv2.split(rgb)[1]
		# rgb = random.choice(cv2.split(rgb))#
	return rgb

class Aug4CSA(object):#Color Space Augment
	number = 8
	trans = [
			cv2.COLOR_RGB2GRAY, cv2.COLOR_RGB2BGR, 
			cv2.COLOR_RGB2XYZ, cv2.COLOR_RGB2LAB,
			cv2.COLOR_RGB2HLS, cv2.COLOR_RGB2LUV,        
			cv2.COLOR_RGB2YCrCb, cv2.COLOR_RGB2YUV,
			]
	# trans_test = [
	#         cv2.COLOR_RGB2GRAY, cv2.COLOR_RGB2BGR, 
	#         cv2.COLOR_RGB2XYZ, cv2.COLOR_RGB2LAB,
	#         ]
	@staticmethod
	def forward_val(pic, flag):
		flag %= Aug4CSA.number
		pic['img'] = random_channel(pic['img'], tran=Aug4CSA.trans[flag])
		return pic
	@staticmethod
	def forward_train(pic):  #random channel mixture
		a = random_channel(pic['img'])
		b = random_channel(pic['img'])
		alpha = random.random()#/2+0.1
		pic['img'] = (alpha*a + (1-alpha)*b).astype(np.uint8)
		return pic
	@staticmethod
	def forward_test(pic):
		pic['csa'] = np.concatenate([random_channel(pic['img'], t) for t in Aug4CSA.trans])
		return pic
#end#


def channels_mixture(pic, index=0):
	r = random_channel(pic).astype(np.float32)
	g = pic[:,:,index].astype(np.float32)
	alpha = random.random()/2
	g = (alpha*r + (1-alpha)*g).astype(np.uint8)
	# pic = cv2.merge([g,g,g])
	pic[:,:,index] = g
	return pic

def random_channel_mix(pic, index=0):
	r = random_channel(pic)
	g = random_channel(pic)
	alpha = random.random()#/2
	print(alpha)
	g = (alpha*r + (1-alpha)*g).astype(np.uint8)
	# pic[:,:,index] = g
	return g


def rand_channel(rgb, tran=None):##cv2.COLOR_RGB2HSV,HSV不好#
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
	if tran==cv2.COLOR_RGB2LAB:
		rgb = cv2.split(rgb)
	elif tran==cv2.COLOR_RGB2XYZ:
		rgb = cv2.split(rgb)
	elif tran==cv2.COLOR_RGB2LUV:
		rgb = cv2.split(rgb)
	elif tran==cv2.COLOR_RGB2HLS:
		rgb = cv2.split(rgb)
	elif tran==cv2.COLOR_RGB2YCrCb:
		rgb = cv2.split(rgb)
	elif tran==cv2.COLOR_RGB2YUV:
		rgb = cv2.split(rgb)
	elif tran==cv2.COLOR_RGB2BGR:
		rgb = cv2.split(rgb)
	if len(rgb)==3:
		rgb = random.choice(rgb)#

	return rgb

if __name__ == '__main__':
	from albumentations import CenterCrop
	tran_crop = CenterCrop(height=584, width=565)
	tran_crop = CenterCrop(height=960, width=999)
	X = 241#569
	Y = 539#221

	i = 0
	nums = ['im0162', 'im0163', 'im0235', 'im0236', 'im0239', 'im0240', 'im0255', 'im0291', 'im0319', 'im0324']
	# img = 'G:/Objects/datasets/seteye/eyeraw/stare/test_rgb/{}.png'.format(nums[i])
	# lab = 'G:/Objects/datasets/seteye/eyeraw/stare/test_lab/{}.ah.png'.format(nums[i])
	# out= 'G:/Objects/expSeg/Peers/SkelCon/STARELOO/loo1{}.png'.format(i)
	
	k = 5
	X = 452
	Y = 491
	img = r'G:\Objects\datasets\seteye\eyeraw\chase\test_rgb/{}_test.jpg'.format(k+21)
	lab = r'G:\Objects\datasets\seteye\eyeraw\chase\test_lab/{}_manual1.png'.format(k+21)
	out= r'G:\Objects\expSeg\Peers\SkelCon\082021chase-SLLdi0.9_csmds0.5halfsim3S512C0.1T0.1-fr\chase_chase_a/a{:02d}.png'.format(k)
	print(img, lab, out)
	img = cv2.imread(img, cv2.IMREAD_COLOR)
	lab = cv2.imread(lab, cv2.IMREAD_GRAYSCALE)
	out = cv2.imread(out, cv2.IMREAD_GRAYSCALE)
	out = tran_crop(image=out)['image']
	step=196
	img = img[X:X+step,Y:Y+step]
	lab = lab[X:X+step,Y:Y+step]
	out = out[X:X+step,Y:Y+step]
	

	# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	mix0 = rand_channel(img.copy(), tran=cv2.COLOR_RGB2XYZ)
	mix1 = rand_channel(img.copy(), tran=cv2.COLOR_RGB2YUV)
	# mix1 = random_channel_mix(img.copy())
	mix2 = rand_channel(img.copy(), tran=cv2.COLOR_RGB2LAB)
	mix3 = rand_channel(img.copy(), tran=cv2.COLOR_RGB2LUV)
	mix4 = rand_channel(img.copy(), tran=cv2.COLOR_RGB2HLS)
	mix5 = rand_channel(img.copy(), tran=cv2.COLOR_RGB2YCrCb)
	# cv2.imwrite('rgb.png', img)
	# cv2.imwrite('lab.png', lab)
	out = (out>127).astype(np.uint8)*255
	tp = (out/255*lab/255)*255
	cmp = np.stack([lab, out, tp.astype(np.uint8)], axis=-1)
	cmp = cv2.cvtColor(cmp, cv2.COLOR_BGR2RGB)
	clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) 
	mix0=clahe.apply(mix0) 
	mix1=clahe.apply(mix1) 
	mix2=clahe.apply(mix2) 
	mix3=clahe.apply(mix3) 
	mix4=clahe.apply(mix4) 
	mix5=clahe.apply(mix5) 
	# cv2.imwrite('mix0.png', mix0)
	# cv2.imwrite('mix1.png', mix1)
	# cv2.imwrite('mix2.png', mix2)
	# cv2.imwrite('mix3.png', mix3)
	# cv2.imwrite('mix4.png', mix4)
	# cv2.imwrite('mix5.png', mix5)
	plt.subplot(231),plt.imshow(mix5)
	plt.subplot(232),plt.imshow(mix0)
	plt.subplot(233),plt.imshow(mix1)
	plt.subplot(234),plt.imshow(mix2)
	plt.subplot(235),plt.imshow(mix3)
	plt.subplot(236),plt.imshow(mix4)

	# plt.subplot(121),plt.imshow(img)
	# plt.subplot(122),plt.imshow(cmp)
	plt.show()

class SplitGreen(object):
	def __call__(self, pic):
		pic['img'] =  cv2.split(pic['img'])[1]
		return pic
class SplitGray(object):
	def __call__(self, pic):
		pic['img'] =  cv2.cvtColor(pic['img'], cv2.COLOR_RGB2GRAY)
		return pic
class SplitGrayRand(object):
	def __call__(self, pic):
		g =  cv2.cvtColor(pic['img'], cv2.COLOR_RGB2GRAY)
		p = random.random()
		pic['img'] =  (random_channel(pic['img'])*p+(1-p)*g).astype(np.uint8)
		return pic
class SplitGreenRand(object):
	def __call__(self, pic):
		g =  random_channel(pic['img'])
		p = random.random()
		pic['img'] =  (cv2.split(pic['img'])[1]*p+(1-p)*g).astype(np.uint8)
		return pic
class SplitRand(object):
	def __call__(self, pic):
		g =  random_channel(pic['img'])
		pic['img'] =  random_channel(pic['img'])
		return pic
class SplitRandRand(object):
	def __call__(self, pic):
		g =  random_channel(pic['img'])
		p = random.random()
		pic['img'] =  (random_channel(pic['img'])*p+(1-p)*g).astype(np.uint8)
		return pic

class ColorSpace(object): #八种色彩空间 cv2.COLOR_RGB2BGR, 
	# ['hls','hsv','lab','luv','xyz','yrb','yuv']
	def __call__(self, pic):
		pic['hls'] =  cv2.cvtColor(pic['img'], cv2.COLOR_RGB2HLS)[:,:,1]
		pic['hsv'] =  cv2.cvtColor(pic['img'], cv2.COLOR_RGB2HSV)[:,:,0]
		pic['la_'] =  cv2.cvtColor(pic['img'], cv2.COLOR_RGB2LAB)[:,:,0]
		pic['luv'] =  cv2.cvtColor(pic['img'], cv2.COLOR_RGB2LUV)[:,:,0]
		pic['xyz'] =  cv2.cvtColor(pic['img'], cv2.COLOR_RGB2XYZ)[:,:,0]
		pic['yrb'] =  cv2.cvtColor(pic['img'], cv2.COLOR_RGB2YCrCb)[:,:,0]
		pic['yuv'] =  cv2.cvtColor(pic['img'], cv2.COLOR_RGB2YUV)[:,:,0]
		pic['img'] = pic['img'][:,:,1]
		return pic
class SplitGreenGray(object):
	def __call__(self, pic):
		pic['aux'] =  cv2.cvtColor(pic['img'], cv2.COLOR_RGB2GRAY)
		pic['img'] =  cv2.split(pic['img'])[1]
		return pic
class SplitGreenRand(object):
	def __call__(self, pic):
		pic['aux'] =  cv2.cvtColor(pic['img'], cv2.COLOR_RGB2GRAY)
		pic['img'] =  random_channel(pic['img'])
		return pic
class SplitRandGray(object):
	def __call__(self, pic):
		pic['aux'] =  random_channel(pic['img'])
		pic['img'] =  cv2.split(pic['img'])[1]
		return pic
class SplitRandom(object):
	def __call__(self, pic):
		pic['aux'] =  random_channel(pic['img'])
		pic['img'] =  random_channel(pic['img'])
		return pic


# class ChannelChoice(object):
#     # YCrCb:  Y、-Cb   YUV:   Y、-V   LAB：   L、-A 
#     # LUV:    L、-U    HSV:   V      HLS：   L
#     def __call__(self, pic):
#         pic['img'] = random_channel(pic['img'])
#         return pic

# class MixtureChannel(object):
#     def __call__(self, pic):
#         ch1 = random_channel(pic['img'])
#         ch2 = random_channel(pic['img'])
#         p = random.random()#/2
#         pic['img'] = (p*ch1+(1-p)*ch2).astype(np.uint8)
#         return pic

# class ChannelGreen(object):
#     def __call__(self, pic):
#         pic['img'] =  cv2.split(pic['img'])[1]
#         return pic
# class ChannelGray(object):
#     def __call__(self, pic):
#         pic['img'] =  cv2.cvtColor(pic['img'], cv2.COLOR_RGB2GRAY)
#         return pic

# class MixtureGreen(object):
#     def __call__(self, pic):
#         g =  cv2.split(pic['img'])[1]
#         c = random_channel(pic['img'])
#         p = random.random()
#         pic['img'] = (p*c+(1-p)*g).astype(np.uint8)
#         return pic

# class Mix4Green(object):
#     def __call__(self, pic):
#         r,g,b = cv2.split(pic['img'])
#         g9 = (0.1*r+0.9*g)
#         g8 = (0.2*r+0.8*g)
#         g7 = (0.3*r+0.7*g)
#         g6 = (0.4*r+0.6*g)
#         pic['img'] = np.stack([g,g9,g8,g7,g6], axis=-1)
#         return pic

# class Mix4Green(object):
#     def __call__(self, pic):
#         _,g,_ = cv2.split(pic['img'])
#         xyz = cv2.cvtColor(pic['img'], cv2.COLOR_RGB2XYZ)
#         x,y,z = cv2.split(xyz)
#         gx = (0.5*x+0.5*g)
#         gy = (0.5*y+0.5*g)
#         gz = (0.5*z+0.5*g)
#         pic['img'] = np.stack([g,x,y,z], axis=-1)
#         return pic
		
# class Cat4Green(object):
#     def __call__(self, pic):
#         _,g,_ = cv2.split(pic['img'])
#         cs = [g]
#         for c in [#cv2.COLOR_RGB2GRAY, cv2.COLOR_RGB2BGR,
#             cv2.COLOR_RGB2XYZ, cv2.COLOR_RGB2HLS, cv2.COLOR_RGB2LUV,        
#             cv2.COLOR_RGB2LAB, cv2.COLOR_RGB2YCrCb, cv2.COLOR_RGB2YUV,
#             ]:
#             c = random_channel(pic['img'], c)
#             cs.append(c)
#         pic['img'] = np.stack(cs, axis=-1)
#         return pic



# class GMixture(object):
#     def __call__(self, pic):
#         _,g,b = cv2.split(pic['img'])
#         xyz = cv2.cvtColor(pic['img'], cv2.COLOR_RGB2XYZ)
#         x,y,z = cv2.split(xyz)
#         chs = [b,x,y,z]
#         ch = random.choice(chs)
#         p = random.random()/2
#         pic['img'] = (p*ch+(1-p)*g).astype(np.uint8)
#         return pic

# class XYZChoice(object):
#     def __call__(self, pic):
#         pic['img'] = cv2.cvtColor(pic['img'], cv2.COLOR_RGB2XYZ)
#         pic['img'] = random.choice(cv2.split(pic['img']))
#         return pic

# class GBXYZMixture(object):
#     def __call__(self, pic):
#         r,g,b = cv2.split(pic['img'])
#         xyz = cv2.cvtColor(pic['img'], cv2.COLOR_RGB2XYZ)
#         x,y,z = cv2.split(xyz)
#         chs = [g,b,x,y,z]#
#         ch1,ch2 = random.sample(chs, 2)
#         p = random.random()#/2
#         pic['img'] = (p*ch1+(1-p)*ch2).astype(np.uint8)
#         return pic
