if __name__ == '__main__':
    from tran import *
else:
    from .tran import *

#视网膜眼底图像增强的新方法_陈萌梦.pdf
# 1.高帽黑帽变换；2.CLAHE；3.二维匹配滤波
# def en_hat(img):
#     kernel = cv2.getStructuringElement(shape=cv2.MORPH_CROSS, ksize=(7, 7))
#     tt = cv2.morphologyEx(src=img, op=cv2.MORPH_TOPHAT, kernel=kernel)
#     bt = cv2.morphologyEx(src=img, op=cv2.MORPH_BLACKHAT, kernel=kernel)
#     return tt - bt + img   

# class ArrayHat(object):
#     def __call__(self, pics):
#         pics['img'] = en_hat(pics['img']) 
#         return pics 
      

def retinex(src, sigma=15, ksize=17):
    src = src.astype(np.float32)/255
    logS = np.log(1+EPS+src)
    logL = np.log(1+EPS+cv2.GaussianBlur(src=src, ksize=(ksize,ksize), sigmaX=sigma))
    ret = np.exp(logS - logL)
    #gain/off方法
    ret = gain(ret, p=1)
    return ret.astype(np.uint8)
class EnRetinex(object):
    def __init__(self, new=False):
        self.new = new
    def __call__(self, pics):
        img = pics['img'].copy()
        if self.new:
            pics['ret'] = retinex(img)
        else:
            pics['img'] = retinex(img)
        return pics   

 
def unSharpMedian(sr):
    bgd = cv2.medianBlur(src=sr, ksize=21).astype(np.float32)
    en = gain(sr-bgd)
    return en
class EnDeLight(object):
    def __init__(self, new=False):
        self.new = new
    def __call__(self, pics):
        if self.new:
            pics['ret'] = unSharpMedian(pics['img'])
        else:
            pics['img'] = unSharpMedian(pics['img'])
        return pics   
#start#  
class EnClahe(object):
    def __init__(self, clip=2, grid=8, keys=['img'], random=False):
        self.random = random
        self.keys = keys
        self.clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    def deal(self, pic):
        if self.random:
            if random.choice([True, False]):
                return self.clahe.apply(pic)
            else:
                return pic
        return self.clahe.apply(pic)
    def __call__(self, pics):
        for key in self.keys:
            if len(pics[key].shape)>2:
                for i in range(pics[key].shape[-1]):
                    pics[key][i] = self.deal(pics[key][i])
            else:
                pics[key] = self.deal(pics[key])
        return pics   
#end#  

class EnGamma(object):
    def __init__(self, gamma=.3):#adjust_gamma0.454
        # build a lookup table mapping the pixel values [0, 255] to the adjusted gamma values
        # 具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
        self.table = np.array([((i/255.0)**gamma)*255 for i in np.arange(0, 256)]).astype(np.uint8)
    def __call__(self, pics):
        # pics['img'] = cv2.LUT(pics['img'], self.table)# apply gamma correction using the lookup table
        if len(pics['img'].shape)>2:
            for i in range(pics['img'].shape[-1]):
                pics['img'][i] = cv2.LUT(pics['img'][i], self.table)
        else:
            pics['img'] = cv2.LUT(pics['img'], self.table)
        return pics

class EnPrePro(object):
    def __init__(self, clip=2, grid=8):
        self.clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    def __call__(self, pics):
        tp = pics['img'].astype(np.float32)
        tp = (tp - tp.mean()) / tp.std()
        # tp = (tp - 0.433) / 0.1024
        tp = gain(tp, .9)
        pics['img'] = self.clahe.apply(tp)
        return pics
