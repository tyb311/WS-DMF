if __name__ == '__main__':
    from tran import *
else:
    from .tran import *


#start#
import pywt

class Wavelet2D(object):
    def calc_shape(self, img):
        shape_pad = False
        h, w = img.shape
        nh = 4 * (h // 4)
        if h != nh:
            shape_pad = True
            nh = 4 * (h // 4 + 1)
        nw = 4 * (w // 4)
        if w != nw:
            shape_pad = True
            nw = 4 * (w // 4 + 1)
        if nh > nw:
            nw = nh
        else:
            nh = nw
        self.shape_raw = img.shape
        self.shape_new = (nh, nw)
        self.shape_off = (nh - h)//2, (nw - w)//2
        return shape_pad
    def _pad(self, img):
        self._shape_pad = self.calc_shape(img)
        if not self._shape_pad:
            return img
        h, w = self.shape_raw
        new_img = np.zeros(self.shape_new)
        oh, ow = self.shape_off
        new_img[oh:oh + h, ow:ow + w] = img
        return new_img
    def _unpad(self, img):
        if not self._shape_pad:
            return img
        h, w = self.shape_raw
        new_img = np.zeros(self.shape_raw)
        oh, ow = self.shape_off
        new_img[:, :] = img[oh:oh + h, ow:ow + w]
        return new_img
    def __call__(self, img):
        wave = pywt.Wavelet('haar')
        cA1, (cH1, cV1, cD1) = pywt.swt2(self._pad(img), wave, level=1, start_level=1)[0]
        return self._unpad(cA1)#pics
#end#
    # def __call__(self, pics):
    #     img = pics['img']

    #     wave = pywt.Wavelet('haar')
    #     cA1, (cH1, cV1, cD1) = pywt.swt2(self._pad(img), wave, level=1, start_level=1)[0]

    #     pics['cA1'] = gain(self._unpad(cA1),1)
    #     pics['cH1'] = gain(self._unpad(cH1),1)
    #     pics['cV1'] = gain(self._unpad(cV1),1)
    #     pics['cD1'] = gain(self._unpad(cD1),1)
    #     return pics

        # cA1 = np.ones_like(cA1)
        # coeffs = ((cA1, (cH1, cV1, cD1)),)
        # swt = pywt.iswt2(coeffs, wave)
        # swt = clip(self._unpad(swt))
        # # swt = gain_off(self._unpad(swt), p=.2)
        # pics['img'] = swt