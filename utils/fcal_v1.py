
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

#start#
import skimage
from skimage import morphology
from skimage import measure
print('Scikit-Image:', skimage.__version__)
class FCAL(object):
    alpha=5
    beta=5
    def __init__(self):
        self.kernel_a = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.alpha, self.alpha))
        self.kernel_b = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.beta, self.beta))

    def forward(self, RefVessels, SrcVessels):
        # % Initialization
        SrcVessels = SrcVessels.round()
        RefVessels = RefVessels.round()
        # print('FCAL:', SrcVessels.shape, RefVessels.shape, self.kernel_a.shape, self.kernel_b.shape)

        # % Calculation of C
        # Lref, Cref = measure.label(RefVessels, 8, return_num=True)
        # Lsrc, Csrc = measure.label(SrcVessels, 8, return_num=True)
        # use 'connectivity' instead. For neighbors=8, use connectivity=1
        Lref, Cref = measure.label(RefVessels, connectivity=1, return_num=True)
        Lsrc, Csrc = measure.label(SrcVessels, connectivity=1, return_num=True)
        C = 1 - min(1, abs(Cref-Csrc)/RefVessels.sum())


        # % Calculation of A
        dilatedRefAlpha = morphology.dilation(RefVessels, self.kernel_a)
        dilatedSrcAlpha = morphology.dilation(SrcVessels, self.kernel_a)
        dilateOverlap = dilatedSrcAlpha * RefVessels + dilatedRefAlpha * SrcVessels
        dilateOverlap[dilateOverlap>0] = 1
        # dilateOverlap = dilateOverlap.round()
        Overlap = RefVessels + SrcVessels
        # Overlap = Overlap.round()
        Overlap[Overlap>0] = 1
        A = dilateOverlap.sum() / Overlap.sum()

        # % Calculation of L
        SrcSkeleton = morphology.skeletonize(SrcVessels, method='zhang')
        RefSkeleton = morphology.skeletonize(RefVessels, method='zhang')

        dilatedRefBeta = morphology.dilation(RefVessels, self.kernel_b)
        dilatedSrcBeta = morphology.dilation(SrcVessels, self.kernel_b)

        dilateSkelOverlap = SrcSkeleton * dilatedRefBeta + RefSkeleton * dilatedSrcBeta
        dilateSkelOverlap[dilateSkelOverlap>0] = 1
        SkelOverlap = SrcSkeleton + RefSkeleton
        SkelOverlap[SkelOverlap>0] = 1
        L = dilateSkelOverlap.sum() / SkelOverlap.sum()

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

        print(i, f, c, a, l)
        fs.append(f)
        cs.append(c)
        aa.append(a)
        ls.append(l)
    print(sum(fs) / len(fs))
    print(sum(cs) / len(cs))
    print(sum(aa) / len(aa))
    print(sum(ls) / len(ls))
    # DRIS	F=0.8494,C=0.9968,A=0.9491,L=0.8974

'''
function [ f,C,A,L ] = CAL( SrcVessels, RefVessels, alpha, beta )
%CAL Summary of this function goes here
%   Detailed explanation goes here

% Initialization
SrcVessels(SrcVessels>0) = 1;
RefVessels(RefVessels>0) = 1;

% Calculation of C
[Lref, Cref] = bwlabel(RefVessels, 8);%L=杩為�鍩熸爣绛撅紝C=杩為�鍩熶釜鏁�
[Lsrc, Csrc] = bwlabel(SrcVessels, 8);
C = 1 - min(1, abs(Cref-Csrc)/sum(sum(RefVessels)));

% Calculation of A
SE = strel('disk', alpha);
dilatedRefAlpha = imdilate(RefVessels,SE);
dilatedSrcAlpha = imdilate(SrcVessels,SE);
dilateOverlap = dilatedSrcAlpha .* RefVessels + dilatedRefAlpha .* SrcVessels;
dilateOverlap(dilateOverlap>0) = 1;
Overlap = RefVessels + SrcVessels;
Overlap(Overlap>0) = 1;
A = sum(sum(dilateOverlap)) / sum(sum(Overlap));

% Calculation of L
SrcSkeleton = uint8(bwmorph(SrcVessels,'thin',inf));
RefSkeleton = uint8(bwmorph(RefVessels,'thin',inf));

SE = strel('disk', beta);
dilatedRefBeta = imdilate(RefVessels,SE);
dilatedSrcBeta = imdilate(SrcVessels,SE);

dilateSkelOverlap = SrcSkeleton .* dilatedRefBeta + RefSkeleton .* dilatedSrcBeta;
dilateSkelOverlap(dilateSkelOverlap>0) = 1;
SkelOverlap = SrcSkeleton + RefSkeleton;
SkelOverlap(SkelOverlap>0) = 1;
L = sum(sum(dilateSkelOverlap)) / sum(sum(SkelOverlap));

% Score of the CAL function
f = C * A * L;

end
'''