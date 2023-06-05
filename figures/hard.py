import cv2, os
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

import numpy as np
from sklearn import metrics
from albumentations import PadIfNeeded
from skimage import morphology
from fcal import fcal

folder_rgba = r'G:\Objects\datasets\seteye\eyeraw\drive\test_rgb'
folder_pred = r'G:\Objects\HisEye\Exp07h\072415drive-siamXsunetXsunet-fr\drive_drive_los'
folder_true = r'G:\Objects\datasets\seteye\drive\test_lab'
folder_dris = r'G:\Objects\HisEye\EyeSOTA\DRIS_code\DRIVE\output_RES_DRIVE_STANDARD'

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
def skel_iou(RefVessels, SrcVessels):
	SrcVessels = SrcVessels.round()
	RefVessels = RefVessels.round()
	# print('FCAL:', SrcVessels.shape, RefVessels.shape, self.kernel.shape, self.kernel.shape)

	# % Calculation of L
	SrcSkeleton = morphology.skeletonize(SrcVessels, method='zhang')
	RefSkeleton = morphology.skeletonize(RefVessels, method='zhang')
	RefSD = morphology.dilation(RefSkeleton, kernel)
	SrcSD = morphology.dilation(SrcSkeleton, kernel)
	inter = (RefSD * SrcSD).sum()
	outer = (RefSD + SrcSD).sum()
	C = inter / outer
	return C


import shutil
FOLDER = 'G:/Objects/SegCL/figures/hard'
if os.path.exists(FOLDER):
	# os.removedirs(FOLDER)
	shutil.rmtree(FOLDER)
	os.mkdir(FOLDER)


def analysis(idx):
	rgba = os.path.join(folder_rgba, '{:02d}_test.tif'.format(idx+1))
	true = os.path.join(folder_true, '{:02d}_lab1.png'.format(idx+1))
	pred = os.path.join(folder_pred, 'los{:02d}.png'.format(idx))
	dris = os.path.join(folder_dris, '{:02d}_test.png'.format(idx+1))

	rgba = cv2.imread(rgba, cv2.IMREAD_COLOR)
	rgba = cv2.cvtColor(rgba, cv2.COLOR_BGR2RGB)
	true = cv2.imread(true, cv2.IMREAD_GRAYSCALE)
	pred = cv2.imread(pred, cv2.IMREAD_GRAYSCALE)
	dris = cv2.imread(dris, cv2.IMREAD_GRAYSCALE)

	if true.shape!=pred.shape or true.shape!=dris.shape:
		# print(true.shape, pred.shape, dris.shape)
		h, w = true.shape

		# padh, padw = (pred.shape[0]-true.shape[0])//2, (pred.shape[1]-true.shape[1])//2
		# pred = pred[:h, :w]
		# pred = pred[padh:padh+h, padw:padw+w]

		divide = 32
		imgw = int(np.ceil(w / divide)) * divide
		imgh = int(np.ceil(h / divide)) * divide
		augPad = PadIfNeeded(p=1, min_height=imgh, min_width=imgw)
		true = augPad(image=true)['image']
		dris = augPad(image=dris)['image']
		rgba = augPad(image=rgba)['image']
		# print(true.shape, pred.shape, dris.shape)
		# assert true.shape==pred.shape, 'shape not match'+str(pred.shape)+str(true.shape)
		# print(true.shape, pred.shape)

	# print(pred.min(), pred.max())
	# print(true.min(), true.max())
	# zero = np.zeros(size=(h,w,3), dtype=np.float32)
	zero = np.zeros_like(pred)
	prob = np.stack([true, pred, zero], axis=-1)
	# _,otsu = cv2.threshold(pred, 127, 255, cv2.THRESH_OTSU)
	thsh = (pred>127).astype(np.uint8)*255
	thsh = morphology.dilation(thsh, selem=np.ones(shape=(5,5), dtype=np.float32))
	diff = true.copy()
	diff[thsh>0] = 0


	# bpr = (thsh>127).astype(np.uint8)
	# bgt = (true>127).astype(np.uint8)
	# bdr = (dris>127).astype(np.uint8)
	# ##  hard sample mining
	# crop = 64
	# stride = crop#crop//2
	# for h in range(0, true.shape[0]-crop, stride):
	# 	for w in range(0, true.shape[1]-crop, stride):
	# 		# iou = metric_iou(pred[h:h+crop, w:w+crop], true[h:h+crop, w:w+crop])
	# 		gt = bgt[h:h+crop, w:w+crop]#.reshape(-1)
	# 		pr = bpr[h:h+crop, w:w+crop]#.reshape(-1)
	# 		# print(pr.min(), pr.max())
	# 		# print(gt.min(), gt.max())
	# 		# if gt.sum()>9 and pr.sum()>9 and (pr*gt).sum()>9:
			
	# 		if gt.sum()>9:
	# 			# 局部分割错误与否不是看交并比，而是错误像素数
	# 			iou = np.round(skel_iou(gt, pr), 2)
	# 			# eer = (gt != pr).sum()# / gt.shape[0]**2
	# 			# f,c,a,l = fcal(pr, gt)
	# 			eer = (gt.astype(np.float32) - pr.astype(np.float32) > 0).sum()


	# 			color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
	# 			cv2.putText(diff, str(np.round(iou,2)), (h,w+crop), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.6, color=255)

	# 			# if eer>5:#
	# 			if iou<0.7:
	# 				save = cv2.cvtColor(diff[h:h+crop, w:w+crop], cv2.COLOR_BGR2RGB)
	# 				cv2.imwrite(f'{FOLDER}/{h}_{w}.png', save)
	# 			# if iou<0.5:
	# 				# print(h,w,iou)
	# 				print(h,w,eer, gt.shape[0])
	# 				cv2.rectangle(diff, (h,w), (h+crop,w+crop), color)
	# 				cv2.rectangle(prob, (h,w), (h+crop,w+crop), color, 1)
	# 				# cv2.putText(diff, str(eer), (h,w+crop), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.6, color=color)
	# 				# break


	##  hard sample mining
	diff = morphology.dilation(diff, selem=np.ones(shape=(5,5), dtype=np.float32))
	bina = (diff>0).astype(np.uint8)
	#####################################################################
	# #地毯式搜索
	#####################################################################
	crop = 64
	stride = crop*3//4
	for h in range(0, true.shape[0]-crop+1, stride):
		for w in range(0, true.shape[1]-crop+1, stride):
			# iou = metric_iou(pred[h:h+crop, w:w+crop], true[h:h+crop, w:w+crop])
			area = np.round(bina[h:h+crop, w:w+crop].sum(), 2)
			cv2.putText(diff, str(area), (w, h+crop), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.6, color=255)

			if area>99:
			# if w*h>32:
			# if len(contours[j])>19:
				color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
				cv2.rectangle(diff, (w,h), (w+crop, h+crop), 255, 1)
				cv2.rectangle(rgba, (w,h), (w+crop, h+crop), color, 1)
				cv2.rectangle(prob, (w,h), (w+crop, h+crop), color, 1)
			# 直接在图片上进行绘制，所以一般要将原图复制一份，再进行绘制
			# cv2.rectangle(diff, (x, y), (x + w, y + h), (0, 0, 255), 2)

				# save = cv2.cvtColor(diff[y:y-32, x:x+32], cv2.COLOR_BGR2RGB)
				# save = diff[y-32:y+32, x-32:x+32]
				# # print(save.shape, save.dtype)
				# cv2.imwrite(f'{FOLDER}/diff_{h}_{w}.png', save)
				# save = rgba[y-32:y+32, x-32:x+32]
				# cv2.imwrite(f'{FOLDER}/rgba_{h}_{w}.png', save)
	#####################################################################


	#####################################################################
	# #遍历边缘面积
	#####################################################################
	# contours, hierarchy = cv2.findContours(diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	# print('countours:', len(contours))
	# for j in range(len(contours)):
	# 	color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
	# 	# 外接图形
	# 	x, y, w, h = cv2.boundingRect(contours[j])
	# 	if x<32 or x>imgw-32 or y<32 or y>imgh-32:
	# 		continue

	# 	area = cv2.contourArea(contours[j])
	# 	cv2.drawContours(diff, contours[j], 0, 255, -1)#绘制轮廓，填充

	# 	cv2.putText(diff, str(area), (x, y), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.6, color=255)
	# 	# cv2.putText(diff, str(len(contours[j])), (y,x), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.6, color=255)
	# 	# cv2.putText(diff, str(w*h), (y,x), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.6, color=255)
	# 	# print(w, h, w*h, len(contours[j]))

		# if area>49:
		# # if w*h>32:
		# # if len(contours[j])>19:
		# 	cv2.rectangle(diff, (x-32, y-32), (x+32, y+32), 255, 1)
		# 	cv2.rectangle(rgba, (x-32, y-32), (x+32, y+32), color, 1)
		# 	cv2.rectangle(prob, (x-32, y-32), (x+32, y+32), color, 1)
		# 	# cv2.rectangle(diff, (y-32, x-32), (y+32, x+32), 255, 1)
		# 	# cv2.rectangle(rgba, (y-32, x-32), (y+32, x+32), color, 1)
		# 	# cv2.rectangle(prob, (y-32, x-32), (y+32, x+32), color, 1)
		# # 直接在图片上进行绘制，所以一般要将原图复制一份，再进行绘制
		# # cv2.rectangle(diff, (x, y), (x + w, y + h), (0, 0, 255), 2)

		# 	# save = cv2.cvtColor(diff[y:y-32, x:x+32], cv2.COLOR_BGR2RGB)
		# 	save = diff[y-32:y+32, x-32:x+32]
		# 	# print(save.shape, save.dtype)
		# 	cv2.imwrite(f'{FOLDER}/diff_{h}_{w}.png', save)
		# 	save = rgba[y-32:y+32, x-32:x+32]
		# 	cv2.imwrite(f'{FOLDER}/rgba_{h}_{w}.png', save)
	#####################################################################


	plt.subplot(131),plt.imshow(diff)
	plt.subplot(132),plt.imshow(prob)
	plt.subplot(133),plt.imshow(rgba)
	# plt.subplot(133),plt.imshow(np.stack([true, dris, zero], axis=-1))
	plt.suptitle('red:true, green:pred')

	# plt.imshow(diff)
	plt.show()




analysis(0)


for idx in range(20):
	analysis(idx)

# print(sum(ious) / len(ious))