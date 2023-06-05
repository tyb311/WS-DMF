import torch
import torch.nn as nn

#start#
from torchvision.models import vgg16
class PerceptionLoss(nn.Module):
    def __init__(self, device=torch.device('cpu'), layer=31):
        super(PerceptionLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:layer]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network.to(device)
        self.mse_loss = nn.MSELoss()

    def forward(self, pr, gt):
        pr = torch.cat([pr,pr,pr], dim=1)
        gt = torch.cat([gt,gt,gt], dim=1)
        return self.mse_loss(self.loss_network(pr), self.loss_network(gt))

def gram_matrix(y):
	(b,ch,h,w) = y.size() # 比如1,8,2,2
	features = y.view(b,ch,w*h)  # 得到1,8,4
	features_t = features.transpose(1,2) # 调换第二维和第三维的顺序，即矩阵的转置，得到1,4,8
	gram = features.bmm(features_t)# / (ch*h*w) # bmm()用来做矩阵乘法，及未转置的矩阵乘以转置后的矩阵，得到的就是1,8,8了
    # 由于要对batch中的每一个样本都计算Gram Matrix，因此使用bmm()来计算矩阵乘法，而不是mm()
	return gram

from torchvision.models import vgg16
class TextureLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(TextureLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network.to(device)
        self.mse_loss = nn.MSELoss()

    def forward(self, pr, gt):
        pr = torch.cat([pr,pr,pr], dim=1)
        gt = torch.cat([gt,gt,gt], dim=1)
        perception_loss = self.mse_loss(gram_matrix(self.loss_network(pr)), gram_matrix(self.loss_network(gt)))
        return perception_loss

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
    @staticmethod
    def binary_focal(pr, gt, fov=None, gamma=2, *args):
        return -gt     *torch.log(pr)      *torch.pow(1-pr, gamma)
    def forward(self, pr, gt, fov=None, gamma=2, eps=1e-6, *args):
        pr = torch.clamp(pr, eps, 1-eps)
        loss1 = self.binary_focal(pr, gt)
        loss2 = self.binary_focal(1-pr, 1-gt)
        loss = loss1 + loss2
        return loss.mean()

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    @staticmethod
    def binary_cross_entropy(pr, gt, eps=1e-6):#alpha=0.25
        pr = torch.clamp(pr, eps, 1-eps)
        loss1 = -gt     *torch.log(pr) 
        loss2 = -(1-gt) *torch.log((1-pr))   
        return loss1, loss2 
        
    def forward(self, pr, gt, eps=1e-6, *args):#alpha=0.25
        loss1, loss2 = self.binary_cross_entropy(pr, gt) 
        return (loss1 + loss2).mean()#.item()

class DiceLoss(nn.Module):
    __name__ = 'DiceLoss'
    # DSC(A, B) = 2 * |A ^ B | / ( | A|+|B|)
    def __init__(self, ):
        super(DiceLoss, self).__init__()
        self.func = self.dice
    def forward(self, pr, gt, **args):
        return 2-self.dice(pr,gt)-self.dice(1-pr,1-gt)
        # return 1-self.func(pr, gt)
    @staticmethod
    def dice(pr, gt, smooth=1):#self, 
        pr,gt = pr.view(-1),gt.view(-1)
        inter = (pr*gt).sum()
        union = (pr+gt).sum()
        return (smooth + 2*inter) / (smooth + union)#*0.1

class FusionLoss(nn.Module):
    def __init__(self, *args):
        super(FusionLoss, self).__init__()
        self.losses = nn.ModuleList([*args])
    def forward(self, pr, gt):
        return sum([m(pr, gt) for m in self.losses])

def get_loss(mode='fr'):
    print('loss:', mode)
    if mode=='fr':
        return FocalLoss()
    elif mode=='ce':
        return BCELoss()
    elif mode=='di':
        return DiceLoss()
    elif mode=='l2':
        return nn.MSELoss(reduction='mean')
        
    elif mode=='fd':
        return FusionLoss(FocalLoss(), DiceLoss())
    elif mode=='cd':
        return FusionLoss(BCELoss(), DiceLoss())
    elif mode=='1d':
        return FusionLoss(nn.L1Loss(reduction='mean'), DiceLoss())
    elif mode=='2d':
        return FusionLoss(nn.MSELoss(reduction='mean'), DiceLoss())
        
    elif mode=='frp':
        return FusionLoss(FocalLoss(), PerceptionLoss())
    elif mode=='frt':
        return FusionLoss(FocalLoss(), TextureLoss())
    else:
        raise NotImplementedError()
#end#


if __name__ == '__main__':
    x = torch.rand(5, 1, 64, 64)
    y = torch.rand(5, 1, 64, 64).round()

    l = get_loss('fr')(x, y)
    print(l.item())
    l = get_loss('frp')(x, y)
    print(l.item())
    l = get_loss('frt')(x, y)
    print(l.item())