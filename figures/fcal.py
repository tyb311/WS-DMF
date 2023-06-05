from skimage import morphology
from skimage import measure
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def diskStrel(radius):
    sel = np.ones(shape=(2*radius + 1, 2*radius + 1), dtype=np.uint8)

    borderWidth = 0
    if radius == 1: borderWidth = 0
    elif radius == 3: borderWidth = 0
    elif radius == 5: borderWidth = 2
    elif radius == 7: borderWidth = 2
    elif radius == 9: borderWidth = 4
    elif radius == 11: borderWidth = 6
    elif radius == 13: borderWidth = 6
    elif radius == 15: borderWidth = 8
    elif radius == 17: borderWidth = 8
    elif radius == 19: borderWidth = 10
    elif radius == 21: borderWidth = 10
    else: borderWidth = 2

    for i in range(borderWidth):
        for j in range(borderWidth - i):
            sel[i, j] = 0
            sel[i, sel.shape[1] - 1 - j] = 0
            sel[sel.shape[0] - 1 - i, j] = 0
            sel[sel.shape[0] - 1 - i, sel.shape[1] - 1 - j] = 0
    # print(sel)
    return sel
# ————————————————
# 版权声明：本文为CSDN博主「chensonglu」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/lkj345/article/details/58133272


def fcal(SrcVessels, RefVessels, alpha=5, beta=5):
        
    # % Initialization
    # SrcVessels[SrcVessels>0] = 1
    # RefVessels[RefVessels>0] = 1
    SrcVessels = SrcVessels.round()
    RefVessels = RefVessels.round()


    # % Calculation of C
    # [Lref, Cref] = bwlabel(RefVessels, 8)%L=杩為�鍩熸爣绛撅紝C=杩為�鍩熶釜鏁�
    # [Lsrc, Csrc] = bwlabel(SrcVessels, 8)
    # C = 1 - min(1, abs(Cref-Csrc)/sum(sum(RefVessels)))
    # Lref, Cref = measure.label(RefVessels, 8, return_num=True)
    # Lsrc, Csrc = measure.label(SrcVessels, 8, return_num=True)
    Lref, Cref = measure.label(RefVessels, connectivity=2, return_num=True)
    Lsrc, Csrc = measure.label(SrcVessels, connectivity=2, return_num=True)
    C = 1 - min(1, abs(Cref-Csrc)/RefVessels.sum())


    # % Calculation of A
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (alpha, alpha))
    kernel = diskStrel(2)
    dilatedRefAlpha = morphology.dilation(RefVessels,kernel)
    dilatedSrcAlpha = morphology.dilation(SrcVessels,kernel)
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

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (beta, beta))
    kernel = diskStrel(2)
    dilatedRefBeta = morphology.dilation(RefVessels,kernel)
    dilatedSrcBeta = morphology.dilation(SrcVessels,kernel)

    dilateSkelOverlap = SrcSkeleton * dilatedRefBeta + RefSkeleton * dilatedSrcBeta
    dilateSkelOverlap[dilateSkelOverlap>0] = 1
    # dilateSkelOverlap = dilateSkelOverlap.round()
    SkelOverlap = SrcSkeleton + RefSkeleton
    SkelOverlap[SkelOverlap>0] = 1
    # SkelOverlap = SkelOverlap.round()
    L = dilateSkelOverlap.sum() / SkelOverlap.sum()

    return C*A*L, C, A, L



if __name__ == '__main__':
    folder_pred = r'G:\Objects\HisEye\EyeSOTA\DRIS_code\DRIVE\output_RES_DRIVE_STANDARD'
    folder_true = r'G:\Objects\datasets\seteye\eyeraw\drive\test_lab'


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
        # true = cv2.imread(true, cv2.IMREAD_GRAYSCALE)
        # pred = cv2.imread(pred, cv2.IMREAD_GRAYSCALE)

        true = (true>127).astype(np.uint8)
        pred = (pred>127).astype(np.uint8)
        [f, c, a, l] = fcal(pred, true)

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