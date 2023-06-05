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

#start#
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
			score_value = (scores.mean(axis=0)).astype(np.int64)
			print(model_name, score_value)
			record = PdRecord(name=self.json_name, index=self.score_names)
			record.set_list(model_name, score_value)
			record.end()
	
	def desc(self):
		record = PdRecord(name=self.json_name, index=self.score_names)
		record.desc(transpose=True)
#end#