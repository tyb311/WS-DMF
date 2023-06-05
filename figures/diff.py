import cv2, os
import matplotlib.pyplot as plt
import numpy as np

folder_pred = r'G:\Objects\HisEye\Exp07h\072415drive-siamXsunetXsunet-fr\drive_drive_los'
folder_true = r'G:\Objects\datasets\seteye\drive\test_lab'
folder_dris = r'G:\Objects\HisEye\EyeSOTA\DRIS_code\DRIVE\output_RES_DRIVE_STANDARD'

for i in range(20):
    true = os.path.join(folder_true, '{:02d}_lab1.png'.format(i+1))
    pred = os.path.join(folder_pred, 'los{:02d}.png'.format(i))
    dris = os.path.join(folder_dris, '{:02d}_test.png'.format(i+1))

    true = cv2.imread(true, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred, cv2.IMREAD_GRAYSCALE)
    dris = cv2.imread(dris, cv2.IMREAD_GRAYSCALE)

    padh, padw = (pred.shape[0]-true.shape[0])//2, (pred.shape[1]-true.shape[1])//2
    h, w = true.shape
    # pred = pred[:h, :w]
    pred = pred[padh:padh+h, padw:padw+w]
    # assert true.shape==pred.shape, 'shape not match'+str(pred.shape)+str(true.shape)
    # print(true.shape, pred.shape)

    # diff = np.zeros(size=(h,w,3), dtype=np.float32)
    diff = np.zeros_like(pred)
    _,otsu = cv2.threshold(pred, 127, 255, cv2.THRESH_OTSU)
    thsh = (pred>127).astype(np.uint8)*255
    plt.subplot(131),plt.imshow(np.stack([true, otsu, diff], axis=-1))
    plt.subplot(132),plt.imshow(np.stack([true, thsh, diff], axis=-1))
    plt.subplot(133),plt.imshow(np.stack([true, dris, diff], axis=-1))
    plt.title('red:true, green:pred')
    plt.show()