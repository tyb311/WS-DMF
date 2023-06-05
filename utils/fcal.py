
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import skimage
print('Scikit-Image:', skimage.__version__)
#start#
from skimage import morphology
from skimage import measure
class FCAL(object):##   fcal 中a是比较好且稳定的，a>l>f>c
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    def forward(self, RefV, SrcV):
        # % Initialization
        SrcV = SrcV.round()
        RefV = RefV.round()
        # print('FCAL:', SrcV.shape, RefV.shape, self.kernel.shape, self.kernel.shape)
        C,A,L = 1,1,1
        # SrcS = morphology.skeletonize(SrcV, method='zhang').astype(np.uint8)
        # RefS = morphology.skeletonize(RefV, method='zhang').astype(np.uint8)

        RefD = morphology.dilation(RefV, self.kernel)
        SrcD = morphology.dilation(SrcV, self.kernel)
        # print(RefD.min(), RefD.max(), SrcD.min(), SrcD.max())

        # % Calculation of L
        # dilateOverlap = SrcD * RefV + RefD * SrcV
        # interAera = dilateOverlap.sum()
        # L = interAera / (RefV + SrcV).sum()

        # % Calculation of A
        dilateOverlap = SrcD * RefV + RefD * SrcV
        dilateOverlap[dilateOverlap>0] = 1
        Overlap = RefV + SrcV
        Overlap[Overlap>0] = 1
        A = dilateOverlap.sum() / Overlap.sum()

        # # % Calculation of C
        # RefSD = morphology.dilation(RefS, self.kernel)
        # SrcSD = morphology.dilation(SrcS, self.kernel)
        # # print(RefS.min(), RefS.max(), SrcS.min(), SrcS.max())
        # # print(RefSD.min(), RefSD.max(), SrcSD.min(), SrcSD.max())
        # inter = (RefSD * SrcSD).sum()
        # outer = (RefSD + SrcSD).sum() - inter
        # C = inter / outer
        
        # return D, C, A, L
        return C*A*L, C, A, L
#end#


if __name__ == '__main__':
    folder_pred = r'G:\Objects\HisEye\EyeSOTA\DRIS_code\DRIVE\output_RES_DRIVE_STANDARD'
    folder_true = r'G:\Objects\datasets\seteye\eyeraw\drive\test_lab'
    fcal = FCAL()

    fs = []
    cs = []
    aa = []
    ls = []
    for i in range(20):
        pred = folder_pred+'/{:02d}_test.png'.format(i+1)
        true = folder_true+'/{:02d}_manual1.gif'.format(i+1)
        # print(pred, true)

        true = np.array(Image.open(true).convert('L'))
        pred = np.array(Image.open(pred).convert('L'))
        # true = np.random.randint(0, 255, size=true.shape)
        # pred = np.random.randint(0, 255, size=pred.shape)
        # true = cv2.imread(true, cv2.IMREAD_GRAYSCALE)
        # pred = cv2.imread(pred, cv2.IMREAD_GRAYSCALE)

        true = (true>127).astype(np.uint8)
        pred = (pred>127).astype(np.uint8)
        [f, c, a, l] = fcal.forward(true, pred)

        # print(i, f, c, a, l)
        fs.append(f)
        cs.append(c)
        aa.append(a)
        ls.append(l)
    print(sum(fs) / len(fs))
    print(sum(cs) / len(cs))
    print(sum(aa) / len(aa))
    print(sum(ls) / len(ls))
    # DRIS	F=0.8494,C=0.9968,A=0.9491,L=0.8974
