import cv2, os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from albumentations import PadIfNeeded
from skimage import morphology
from fcal import fcal

folder_rgba = r'G:\Objects\datasets\seteye\eyeraw\drive\test_rgb'
folder_pred = r'G:\Objects\HisEye\Exp07h\072415drive-siamXsunetXsunet-fr\drive_drive_los'
folder_true = r'G:\Objects\datasets\seteye\drive\test_lab'
folder_dris = r'G:\Objects\HisEye\EyeSOTA\DRIS_code\DRIVE\output_RES_DRIVE_STANDARD'

# def metric_iou(pr, gt):
# 	pr = pr.round().astype(np.uint8)
# 	gt = gt.round().astype(np.uint8)
# 	outer = (pr + gt).sum()
# 	if outer<9:
# 		return 1
# 	inter = (pr & gt).sum()
# 	iou = inter / (outer - inter + 1)
# 	return np.round(iou, 2)

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
		w = int(np.ceil(w / divide)) * divide
		h = int(np.ceil(h / divide)) * divide
		augPad = PadIfNeeded(p=1, min_height=h, min_width=w)
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
	# _,otsu = cv2.threshold(pred, 127, 255, cv2.THRESH_OTSU)
	thsh = (pred>127).astype(np.uint8)*255
	thsh = morphology.dilation(thsh, selem=np.ones(shape=(5,5), dtype=np.float32))
	diff = np.stack([true, thsh, zero], axis=-1)
	diff[thsh>0] = 0

	bpr = (thsh>127).astype(np.uint8)
	bgt = (true>127).astype(np.uint8)
	bdr = (dris>127).astype(np.uint8)
	##  hard sample mining
	crop = 64
	stride = crop#crop//2
	for h in range(0, true.shape[0]-crop, stride):
		for w in range(0, true.shape[1]-crop, stride):
			# iou = metric_iou(pred[h:h+crop, w:w+crop], true[h:h+crop, w:w+crop])
			gt = bgt[h:h+crop, w:w+crop]#.reshape(-1)
			pr = bpr[h:h+crop, w:w+crop]#.reshape(-1)
			# print(pr.min(), pr.max())
			# print(gt.min(), gt.max())
			# if gt.sum()>9 and pr.sum()>9 and (pr*gt).sum()>9:
			
			if gt.sum()>9:
				# 局部分割错误与否不是看交并比，而是错误像素数
				iou = np.round(metrics.jaccard_score(gt.reshape(-1), pr.reshape(-1), zero_division=1), 2)
				# eer = (gt != pr).sum()# / gt.shape[0]**2
				# f,c,a,l = fcal(pr, gt)
				eer = (gt.astype(np.float32) - pr.astype(np.float32) > 0).sum()


				color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
				cv2.putText(diff, str(np.round(eer,2)), (h,w+crop), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.6, color=color)

				if eer>5:# and iou<0.8
					save = cv2.cvtColor(diff[h:h+crop, w:w+crop], cv2.COLOR_BGR2RGB)
					cv2.imwrite(f'{FOLDER}/{h}_{w}.png', save)
				# if iou<0.5:
					# print(h,w,iou)
					print(h,w,eer, gt.shape[0])
					cv2.rectangle(diff, (h,w), (h+crop,w+crop), color)
					# cv2.putText(diff, str(eer), (h,w+crop), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.6, color=color)
					# break

	
	# print(bpr.min(), bpr.max())
	# print(bgt.min(), bgt.max())
	iou = np.round(metrics.jaccard_score(bgt.reshape(-1), bpr.reshape(-1), zero_division=1), 2)
	# iou = metrics.f1_score(bgt.reshape(-1), bpr.reshape(-1), average='binary', zero_division=0)

	print(idx, iou)
	# plt.subplot(131),plt.imshow(diff)
	# plt.subplot(132),plt.imshow(np.stack([true, thsh, zero], axis=-1))
	# plt.subplot(133),plt.imshow(rgba)
	# # plt.subplot(133),plt.imshow(np.stack([true, dris, zero], axis=-1))
	# plt.suptitle('red:true, green:pred, iou='+str(iou))

	plt.imshow(diff)
	plt.show()

	return iou



analysis(0)


# ious = []
# for idx in range(20):
# 	iou = analysis(idx)
# 	ious.append(iou)

# print(sum(ious) / len(ious))