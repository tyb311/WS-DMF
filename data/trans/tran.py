
# tranDemo = transforms.Compose([
    # ToPILImage(),
    # ToTensor(),
    # ToNorm(mean=[0], std=[1]),
    # # Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    # ])
# Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])),
# Lambda(lambda crops: torch.stack([Normalize([0.485], [0.229])(crop) for crop in crops])),
        
#start#
# 常用资源库
import pandas as pd
import numpy as np
EPS = 1e-6#np.spacing(1)#
import os,glob,numbers

# 图像处理
import math,cv2,random, socket
from PIL import Image, ImageFile, ImageOps, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 图像显示
import matplotlib as mpl
if 'TAN' not in socket.gethostname():
    print('Run on Server!!!')
    mpl.use('Agg')#服务器绘图
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as f
#end#
# from .init import *

def gain(ret, p=1):    #gain_off
    mean = np.mean(ret)
    ret_min = mean-(mean-np.min(ret))*p
    ret_max = mean+(np.max(ret)-mean)*p
    ret = 255*(ret - ret_min)/(ret_max - ret_min)
    ret = np.clip(ret, 0, 255).astype(np.uint8)
    return ret

class TranNull(object):
    def __call__(self, pic):
        return pic

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

def arr2tensor(pic):#, np.float32, copy=False
    if len(pic.shape)==3 and pic.shape[-1]==3:
        pic = pic.transpose((2, 0, 1))
    pic = torch.from_numpy(np.array(pic, np.float32, copy=False))
    return pic.type(torch.float32)

    
from skimage import morphology
class AuxSkeleton(object):
    kernel=cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3,3))
    @staticmethod
    def forward(pic):
        lab_ = (pic>127).astype(np.uint8)#*255
        lab = morphology.skeletonize(lab_)
        lab = morphology.dilation(lab, selem=AuxSkeleton.kernel).astype(np.uint8)
        # print('AuxSkeleton:', lab.max(), lab_.max())
        lab = cv2.bitwise_or(lab_, lab)
        return lab.astype(np.uint8)*255
    def __call__(self, pic):
        pic['aux'] = AuxSkeleton.forward(pic['lab'])
        return pic

def master2tensor(pic):
    _pic=dict()
    if isinstance(pic, dict):
        for key in pic.keys():
            _pic[key] = arr2tensor(pic[key])
        return _pic
    for key in pic[0].keys():
        _pic[key] = torch.stack([arr2tensor(p[key]).squeeze() for p in pic])
    return _pic

def tensor_norm(pic, div=True, uniform=True): 
    for key in pic.keys():
        pic[key] = pic[key].type(torch.float32)
        if div:
            pic[key] = pic[key].div(255)
    #   α and λ denote the mean and standard deviation of the whole dataset
    #   drive mean=0.43296451196074487 std=0.10242566913366317
    # if uniform:
    #     pic['img'] -= pic['img'].mean()
    #     pic['img'] /= pic['img'].std()
    # else:
    #     pic['img'] -=  pic['img'].min()
    #     pic['img'] /=  pic['img'].max()
    return pic


from skimage import morphology
class AuxEdge(object):
    struct=np.ones((5,5))
    def __call__(self, pic):
        # edge = cv2.morphologyEx(src=pic['lab'], op=cv2.MORPH_GRADIENT, kernel=self.kernel)
        # edge1 = cv2.dilate(pic['lab'], self.kernel) - cv2.erode(pic['lab'], self.kernel)
        # print(np.all(edge == edge1))
        att = (pic['lab']>127).astype(np.uint8)
        # edge = morphology.dilation(att, selem=np.ones((5,5)))
        edge = 9*morphology.dilation(att, self.struct)-9*morphology.erosion(att, self.struct)+1
        pic['aux'] = edge/edge.max()*255
        return pic
def fov_clear(pic, value=255, inverse=True):
    fovIndex = (pic['fov']<127).astype(np.bool)
    pic['img'][fovIndex] = value
    if reversed:
        pic['img'] = 255-pic['img']
        pic['red'] = 255-pic['red']
    return pic


class LabReStore(object):
    kernel=cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3,3))
    def __call__(self, pic):
        lab = (pic['lab']>127).astype(np.uint8)
        lab = morphology.skeletonize(lab)
        lab = morphology.dilation(lab, selem=np.ones((3,3)))
        lab = cv2.bitwise_or((pic['lab']>127).astype(np.uint8), (lab>127).astype(np.uint8)).astype(np.uint8)
        pic['lab'] = lab*255#morphology.closing(lab, selem=np.ones((3,3)))
        return pic


def clip(img, length=24):
    mean = img.mean()
    ret_min = mean-length/2
    img = (img.astype(np.float32)-ret_min)/length
    img = np.clip(img, 0, 1)
    return (img*255).astype(np.uint8)
    