
import random, torch
# print(torch.__version__)
# from torch.utils.tensorboard import SummaryWriter

#start#
import copy, torch, torchvision, os
def tensorboard_model(model=None, root='./Alogs', nrow=4):
	net = copy.deepcopy(model).cpu()
	for name, item in net.cpu().named_parameters():
		if not item.requires_grad:
			continue
		if len(item.shape)==4 and item.shape[-1]>3 and item.shape[-2]>3:#只显示四维、尺寸大于3的大核
			tag = os.path.join(root, net.__name__+'_'+name+'.png')
			try:
				item = item.view(-1,1,item.shape[-2],item.shape[-1])
				torchvision.utils.save_image(item, tag, nrow=nrow, normalize=True, scale_each=True)
			except:
				print('tensorboard_model-', name, item.shape, item.dtype)

def tensorboard_logs(logs={}, root='./Alogs', nrow=4):
	for name, item in logs.items():
		if isinstance(item, list) or isinstance(item, tuple):
			continue
		tag = os.path.join(root, name+'.png')
		try:
			item = item.view(-1,1,item.shape[-2],item.shape[-1])
			torchvision.utils.save_image(item, tag, nrow=nrow, normalize=True, scale_each=True)
		except:
			print('tensorboard_logs-', name, item.shape, item.dtype)
#end#
		# self.writer.add_image('cam', torch.randn(size=(1,64,64)), None)
		# self.writer.add_pr_curve('pr_curve', labels=logs['true'], predictions=logs['pred'], global_step=None)
		# self.writer.add_graph(model, torch.ones(size=(1,128,128), dtype=torch.float32))	#This should be available in 1.14 or above.


# 目前可视化指标而已，未来要支持直方图、特征图等描述权重分布
# https://zhuanlan.zhihu.com/p/54947519
#  tensorboard --logdir=Alogs
if __name__ == '__main__':
	y = torch.rand(1, 12, 608, 576)
	from torchvision.models import resnet18
	net = resnet18()
	net.__name__ = 'resnet'
	tensorboard_model(net)
	# tensorboard_logs(logs={'x':y}, root='./')
	# y = torchvision.utils.make_grid(x, nrow=4)
	# print(y.shape, y.min().item(), y.max().item())
	# y = y.data.numpy()

	# torchvision.utils.save_image(y, 'tv.png', normalize=True, scale_each=True)

	# import matplotlib.pyplot as plt
	# plt.imtensorboard(y)
	# plt.tensorboard()
