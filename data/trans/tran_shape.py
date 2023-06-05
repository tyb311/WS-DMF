if __name__ == '__main__':
    from tran import *
else:
    from .tran import *

def pil_rotate(pic, angle):
    _pic=dict()
    for key in pic.keys():
        _pic[key] = pic[key].rotate(angle)
    return _pic

class PILRandomRotation(object):
    def __call__(self, pic):
        angle = random.randint(0,12)*15
        return imgs2arrs(pil_rotate(arrs2imgs(pic), angle))

# class PILFlip(object):
#     def __call__(self, pic):
#         pic = arrs2imgs(pic)
#         tran = random.choice([Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM])
#         _pic=dict()
#         for key in pic.keys():
#             _pic[key] = pic[key].transpose(tran)
#         return imgs2arrs(_pic)

        # tran = random.choice([Image.TRANSPOSE, Image.TRANSVERSE,#正反90度再水平翻转
        #     Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM,
        #     Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270])

#start#
def pil_tran(pic, tran=None):
    if tran is None:
        return pic
    if isinstance(tran, list):
        for t in tran:
            for key in pic.keys():
                pic[key] = pic[key].transpose(t)
    else:
        for key in pic.keys():
            pic[key] = pic[key].transpose(tran)
    return pic

class Aug4Val(object):
    number = 8
    @staticmethod
    def forward_val(pic, flag):
        flag %= Aug4Val.number
        if flag==0:
            return pic
        pic = arrs2imgs(pic)
        if flag==1:
            return imgs2arrs(pil_tran(pic, tran=Image.FLIP_LEFT_RIGHT))
        if flag==2:
            return imgs2arrs(pil_tran(pic, tran=Image.FLIP_TOP_BOTTOM))
        if flag==3:
            return imgs2arrs(pil_tran(pic, tran=Image.ROTATE_180))
        if flag==4:
            return imgs2arrs(pil_tran(pic, tran=[Image.FLIP_LEFT_RIGHT,Image.FLIP_TOP_BOTTOM]))
        if flag==5:
            return imgs2arrs(pil_tran(pic, tran=[Image.ROTATE_180,Image.FLIP_TOP_BOTTOM]))
        if flag==6:
            return imgs2arrs(pil_tran(pic, tran=[Image.ROTATE_180,Image.FLIP_LEFT_RIGHT]))
        if flag==7:
            return imgs2arrs(pil_tran(pic, tran=[Image.ROTATE_180,Image.FLIP_LEFT_RIGHT,Image.FLIP_TOP_BOTTOM]))
#end#



class PILRandomTran(object):#1转为数值，2水平操作
    def __init__(self, horizon=False, vertical=False):
        self.horizon = horizon
        self.vertical = vertical
    def __call__(self, pic):
        pic = arrs2imgs(pic)
        if self.vertical:
            pic = pil_tran(pic, tran=random.choice([None,
                Image.TRANSPOSE, Image.TRANSVERSE,#正反90度再水平翻转
                Image.ROTATE_90, Image.ROTATE_270]))
        if self.horizon:
            pic = pil_tran(pic, tran=random.choice([None,Image.ROTATE_180,
                Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM,]))
        return imgs2arrs(pic)

def random_crops(pic, size=64, num=4):
    def get_box(fov, size):
        rh, rw = fov.shape
        _x = random.randint(0, rw - size)
        _y = random.randint(0, rh - size)
        box = (_x, _y, _x + size, _y + size)
        return box
    def get_crop(pic, size):
        x1,y1,x2,y2 = get_box(pic['lab'], size)
        _pic=dict()
        for key in pic.keys():
            _pic[key] = pic[key][y1:y2,x1:x2]
        return _pic        
    return [get_crop(pic, size) for _ in range(num)]

class RandomAffine(object):
    tran = transforms.RandomAffine(degrees=0, shear=7, resample=Image.NEAREST)
    def __call__(self, pic):
        imgs = np.stack([pic['img'], pic['lab'], pic['fov'], pic['aux']], axis=-1)
        elas = np.array(self.tran(Image.fromarray(imgs)))
        pic['img'], pic['lab'], pic['fov'], pic['aux'] =  elas[..., 0], elas[..., 1], elas[..., 2], elas[..., 3]
        return pic

class RandomScale(object):
    def __call__(self, pic):
        imgs = np.stack([pic['img'], pic['lab'], pic['fov'], pic['aux']], axis=-1)
        h,w = imgs.shape[0:2]
        h = int(h*(1+random.random()/7))
        w = int(w*(1+random.random()/7))
        elas = cv2.resize(src=imgs, dsize=(h,w), interpolation=cv2.INTER_NEAREST)
        pic['img'], pic['lab'], pic['fov'], pic['aux'] =  elas[..., 0], elas[..., 1], elas[..., 2], elas[..., 3]   
        return pic


class RandomPerspective(object):
    tran = transforms.RandomPerspective(distortion_scale=.1, p=1)
    def __call__(self, pic):
        imgs = np.stack([pic['aux'], pic['img'], pic['lab'], pic['fov']], axis=-1)
        elas = np.array(self.tran(Image.fromarray(imgs)))
        pic['aux'], pic['img'], pic['lab'], pic['fov'] =  elas[..., 0], elas[..., 1], elas[..., 2], elas[..., 3]  
        return pic

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

# Function to distort image  alpha = imgs.shape[1]*2、sigma=imgs.shape[1]*0.08、alpha_affine=sigma
def elastic_transform(image, alpha, sigma, alpha_affine):
    random_state = np.random.RandomState(None)
    shape = image.shape
    shape_size = shape[:2]   #(512,512)表示图像的尺寸                
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = 512#min(shape_size) // 3

    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    # Mat getAffineTransform(InputArray src, InputArray dst)  src表示输入的三个点，dst表示输出的三个点，获取变换矩阵M
    M = cv2.getAffineTransform(pts1, pts2)  #获取变换矩阵
    #默认使用 双线性插值，
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)  #构造一个尺寸与dx相同的O矩阵                
    # np.meshgrid 生成网格点坐标矩阵，并在生成的网格点坐标矩阵上加上刚刚的到的dx dy
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))  #网格采样点函数
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return map_coordinates(image, indices, order=3, mode='nearest').reshape(shape)     #order=1,mode='reflect'
       

class RandomElastic(object):
    def __call__(self, pic):
        imgs = np.stack([pic['img'], pic['lab'], pic['fov']], axis=-1)
        elas = elastic_transform(imgs, imgs.shape[1] * 2, imgs.shape[1] * 0.08, imgs.shape[1] * 0.08)
        pic['img'], pic['lab'], pic['fov'] =  elas[..., 0], elas[..., 1], elas[..., 2]   
        return pic

def random_affine(img, targets=None, degrees=(-90, 90), translate=(.1, .1),
            scale=(.9, 1.1), shear=(-2, 2), borderValue=(0,0,0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    border = 0  # width of added border (optional)
    #height = max(img.shape[0], img.shape[1]) + border * 2
    height, width, _ = img.shape 

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue
    return imw
        # pic['img'] = np.array(self.tran(arr2img(pic['img'])))
        # pic['lab'] = np.array(self.tran(arr2img(pic['lab'])))
        # pic['fov'] = np.array(self.tran(arr2img(pic['fov'])))

        # imgs = np.stack([pic['img'], pic['lab'], pic['fov']], axis=-1)
        # elas = random_affine(imgs)
        # pic['img'], pic['lab'], pic['fov'] =  elas[..., 0], elas[..., 1], elas[..., 2]  



def center_crop(pic, size=384):
    pic = arrs2imgs(pic)
    func =  transforms.CenterCrop(size)
    for key in pic.keys():
        pic[key] = func(pic[key])
    return imgs2arrs(pic)  

# class ArrayCrops(object):
#     def __init__(self, size=64, num=4):
#         self.size = size
#         self.num = num
#     def get_box(self, fov):
#         rh, rw = fov.shape
#         _x = random.randint(0, rw - self.size)
#         _y = random.randint(0, rh - self.size)
#         box = (_x, _y, _x + self.size, _y + self.size)
#         return box
#     def get_crop(self, pic):
#         x1,y1,x2,y2 = self.get_box(pic['lab'])
#         _pic=dict()
#         for key in pic.keys():
#             _pic[key] = pic[key][y1:y2,x1:x2]
#         return _pic
#     def __call__(self, pic):
#         return [self.get_crop(pic) for _ in range(self.num)] 

# class PILCenterCrop(object):
#     def __init__(self, size=512):
#         self.size = size
#     def __call__(self, pic):
#         pic = arrs2imgs(pic)
#         pic['img'] = transforms.CenterCrop(self.size)(pic['img'])
#         pic['lab'] = transforms.CenterCrop(self.size)(pic['lab'])
#         pic['fov'] = transforms.CenterCrop(self.size)(pic['fov'])
#         pic['aux'] = transforms.CenterCrop(self.size)(pic['aux'])
#         return imgs2arrs(pic)     

# def expandArray(pic, shape):
#     bgd = np.zeros(shape=shape, dtype=np.uint8)
#     bgd[:pic.shape[0], :pic.shape[1]] = pic
#     return bgd
# class ArrayExpand(object):
#     def __call__(self, pic, divide=32):
#         h, w = pic['fov'].shape
#         w = math.ceil(w / divide) * divide
#         h = math.ceil(h / divide) * divide
#         shape_ = h, w
#         pic['img'] = expandArray(pic['img'], shape=shape_)
#         pic['lab'] = expandArray(pic['lab'], shape=shape_)
#         pic['fov'] = expandArray(pic['fov'], shape=shape_)
#         pic['aux'] = expandArray(pic['aux'], shape=shape_)
#         return pic

# class PILStretch(object):
#     def __call__(self, pic):
#         pic = arrs2imgs(pic)
#         w, h = pic['lab'].size
#         w, h = random.randint(w//2, w), random.randint(h//2, h)
#         pic['img'] = pic['img'].resize((w, h), Image.BILINEAR)
#         pic['lab'] = pic['lab'].resize((w, h), Image.NEAREST)
#         pic['fov'] = pic['fov'].resize((w, h), Image.NEAREST)
#         pic['aux'] = pic['aux'].resize((w, h), Image.BILINEAR)
#         return imgs2arrs(pic)

# class PILScale(object):#param:放缩因子
#     def __call__(self, pic):
#         p = random.random()/2+.5
#         pic = arrs2imgs(pic)
#         w, h = pic['lab'].size
#         w, h = int(w*p),int(h*p)
#         pic['img'] = pic['img'].resize((w, h), Image.BILINEAR)
#         pic['lab'] = pic['lab'].resize((w, h), Image.NEAREST)
#         pic['fov'] = pic['fov'].resize((w, h), Image.NEAREST)
#         pic['aux'] = pic['aux'].resize((w, h), Image.BILINEAR)
#         return imgs2arrs(pic)


if __name__ == '__main__':
    t = transforms.RandomAffine(degrees=0, shear=7, resample=Image.NEAREST)
    x = torch.randint(low=0, high=256, size=(64,64,4)).data.numpy()
    x = Image.fromarray(x.astype(np.uint8))
    y = t(x)
    print(y.size)
    print(np.array(y).shape)
    