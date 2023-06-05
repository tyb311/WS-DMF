
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
from torchvision.transforms import functional as f
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


from hrnet import *
from dou import *
from dmf import *
from lunet import *
from byol import *

#start#
def build_model(model_type='hrdo', arch_type=None, loss_type='nce'):

    if model_type == 'hrdo':
        model = hrdo()
    elif model_type == 'dou':
        model = DoU()
    elif model_type == 'dmf':
        model = dmf32()
    elif model_type.endswith('unet'):
        model = eval(model_type+'()')
    else:
        raise NotImplementedError(f'--> Unknown model_type: {model_type}')

    if arch_type == 'byol':
        model = BYOL(encoder=model, clloss=loss_type)
    elif arch_type == 'siam':
        model = SIAM(encoder=model, clloss=loss_type)
    else:
        print('No this Arch!:'+arch_type)

    return model
#end#


if __name__ == '__main__':
    net = hrdo()
    x = torch.rand(2,1,64,64)
    y = net(x, x)
    print(y.keys())

    # net = build_model('hrdo', '', '')
    net = build_model('lunet', '', '')
    # net = build_model('hrdo', 'siam')
    # net = build_model('hrby', 'simm')
    # net = build_model('hrby', 'nce')#loss=10
    # net = build_model('hrby', 'ncem')#loss=10
    # net = build_model('hrsm', 'sim')
    # net = build_model('hrsm', 'simm')
    # net = build_model('hrsm', 'nce')
    # net = build_model('hrsm', 'ncem')
    net.eval()
    y = net(x, x.round())
    print(y.keys())
    print(net.__name__, y['proj'].shape)

    net.train()
    y = net(x, x.round())
    # emb = net.emb
    # emb = mlp_sample_selection(emb, y['pred'], x.roung(), mode='hazy')
    print(net.__name__, y['loss'].item())


    # plot(net.emb)
