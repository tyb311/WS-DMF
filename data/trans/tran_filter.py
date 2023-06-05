# print(__name__, __file__)
if __name__ == '__main__' or __name__ == 'tran_filter':
    from tran import *
else:
    from .tran import *

#start# 
DIVIDES=12
def kernelsMatch(ksize=13, sigma=1.5, ylens=range(3,8)):#1,10
    halfLength = ksize // 2
    sqrt2pi_sigma = np.sqrt(2 * np.pi) * sigma
    x, y = np.meshgrid(range(ksize), range(ksize))
    filters = []
    for ylen in ylens:
        for theta in np.arange(0, np.pi, np.pi/DIVIDES):
            cos, sin = np.cos(theta), np.sin(theta)
            x_ = (x - halfLength) * cos + (y - halfLength) * sin
            y_ = (y - halfLength) * cos - (x - halfLength) * sin 

            indexZero = np.logical_or(abs(x_) > 3*sigma, abs(y_) > ylen)
            kernel = -np.exp(-0.5 * (x_ / sigma) ** 2) / sqrt2pi_sigma
            kernel[indexZero] = 0
        
            indexFalse = kernel<0
            mean = np.sum(kernel) / np.sum(indexFalse)
            kernel[indexFalse] -= mean
            filters.append(kernel)
    return filters

class MatchFilter(object):
    KERNELS = kernelsMatch()
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(7,7))
    def __init__(self, keys=['img'], new=True, erode=False):
        self.new = new
        self.erode = erode
        self.keys = keys
    @staticmethod
    def apply_filters(img):
        accum = np.zeros_like(img)
        for kern in MatchFilter.KERNELS:
            fimg = cv2.filter2D(img, cv2.CV_8U, kern, borderType=cv2.BORDER_REPLICATE)
            np.maximum(accum, fimg, accum)
        return accum
    def __call__(self, pics):
        for key in self.keys:
            mat = pics[key]#.copy()
            mat = self.apply_filters(mat)
            mat = gain(mat, p=.9)
            if self.erode:
                fov = cv2.erode(src=pics['fov'], kernel=self.kernel, iterations=4)
                mat[fov<128] = 0
            if self.new:
                pics[key+'_mat'] = mat
            else:
                pics[key] = mat
        return pics
#end#
'''

        # pics['aux'] = gain(mat.astype(np.float32)-img.astype(np.float32), p=1)
        if self.use_exp:
            mat = exp1(mat)

# 对数变换增强暗细节，指数变换减弱暗细节,exp1更干净
def exp1(img):
    att = (img.copy().astype(np.float32)/127)**2
    return gain(att)

# def stack_filters(img):
#     accum = []
#     for kern in KERNELS:
#         fimg = cv2.filter2D(img, cv2.CV_8U, kern, borderType=cv2.BORDER_REPLICATE)
#         accum.append(fimg)
#     return np.stack(accum, axis=0)

'''