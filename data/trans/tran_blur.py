if __name__ == '__main__':
    from tran import *
else:
    from .tran import *

#start#
class PILJitter(object):
    tran = transforms.ColorJitter(0.4,0.4,0.4,0.04)
    def __init__(self, keys=['img']):
        self.keys = keys
    def __call__(self, pics):
        for key in self.keys:
            pics[key] = np.array(self.tran(arr2img(pics[key])))
        return pics

class PILGaussianBlur(object):
    def __init__(self, keys=['img']):
        self.keys = keys
    def __call__(self, pics): 
        for key in self.keys:
            tran =ImageFilter.GaussianBlur(radius=1+random.random())
            img = Image.fromarray(pics[key]).filter(tran)
            pics[key] = np.array(img)
        return pics
#end#

def getRandBlurArg():
    return random.choice([
        ImageFilter.BLUR,ImageFilter.MedianFilter(5),
        ImageFilter.DETAIL,ImageFilter.EDGE_ENHANCE,
        ImageFilter.GaussianBlur(radius=1+random.random()),
        ImageFilter.MaxFilter(3),ImageFilter.MinFilter(size=3),
        ImageFilter.SHARPEN,ImageFilter.UnsharpMask])

class PILRandomBlur(object):
    def __init__(self, keys=['img']):
        self.keys = keys
    def __call__(self, pics): 
        for key in self.keys:
            tran = getRandBlurArg()
            img = Image.fromarray(pics[key])
            img = img.filter(tran)
            pics[key] = np.array(img)
        return pics


# class ImageRandomBlur(object):
#     def __call__(self, pics): 
#         tran = getRandBlurArg()
#         pics['img'] = pics['img'].filter(tran)
#         return pics
        
# class BlurMean(object):
#     def __init__(self, ksize=7):
#         self.ksize = (ksize, ksize)
#     def __call__(self, pics): 
#         pics['img'] = cv2.blur(src=pics['img'], ksize=self.ksize)
#         return pics

# class BlurMedian(object):
#     def __init__(self, ksize=7):
#         self.ksize = ksize
#     def __call__(self, pics):
#         # pics['img'] = pics['img'].filter(ImageFilter.BLUR)    
#         pics['img'] = cv2.medianBlur(src=pics['img'], ksize=self.ksize)
#         return pics

# class BlurGaussian(object):
#     def __init__(self, ksize=7):
#         self.ksize = (ksize, ksize)
#     def __call__(self, pics):
#         # pics['img'] = pics['img'].filter(ImageFilter.GaussianBlur(radius=random.random()))  
#         pics['img'] = cv2.GaussianBlur(src=pics['img'], ksize=self.ksize, sigmaX=1)
#         return pics
#end#

if __name__ == '__main__':
    sr = Image.open('img.jpg').convert('L')
    while True:
        pr = ImageRandomBlur()({'img':sr})['img']

        plt.subplot(121),plt.imshow(sr),plt.title('sr')
        plt.subplot(122),plt.imshow(pr),plt.title('pr')
        plt.show()

    # sr = cv2.imread('img.jpg', cv2.IMREAD_COLOR)[:,:,1]
    # print('sr:', sr.shape, sr.dtype)

    # sr_mean = BlurMean()(sr)['img']
    # sr_median = BlurMedian()(sr)['img']
    # sr_gaussian = BlurGaussian()(sr)['img']

    # plt.subplot(221),plt.imshow(sr['img']),plt.title('sr')
    # plt.subplot(222),plt.imshow(sr_mean),plt.title('mean')
    # plt.subplot(223),plt.imshow(sr_median),plt.title('median')
    # plt.subplot(224),plt.imshow(sr_gaussian),plt.title('gaussian')
    # plt.show()