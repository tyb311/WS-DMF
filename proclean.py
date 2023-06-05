import os
import shutil
import torch

# shutil.rmtree(file_name)即可解决
##################################################################
def pyProjClean():
	'''
	os.walk() 方法用于通过在目录树中游走输出在目录中的文件名，向上或者向下。
	os.walk() 方法是一个简单易用的文件、目录遍历器，可以帮助我们高效的处理文件、目录方面的事情。
	在Unix，Windows中有效。

	语法：
		os.walk(top[, topdown=True[, onerror=None[, followlinks=False]]])

	参数：
		top -- 是你所要遍历的目录的地址, 返回的是一个三元组(root,dirs,files)。
			root 所指的是当前正在遍历的这个文件夹的本身的地址
			dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
			files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
		topdown --可选，为 True，则优先遍历 top 目录，否则优先遍历 top 的子目录(默认为开启)。如果 topdown 参数为 True，walk 会遍历top文件夹，与top 文件夹中每一个子目录。
		onerror -- 可选，需要一个 callable 对象，当 walk 需要异常时，会调用。
		followlinks -- 可选，如果为 True，则会遍历目录下的快捷方式(linux 下是软连接 symbolic link )实际所指的目录(默认关闭)，如果为 False，则优先遍历 top 的子目录。
	'''
	for root, dirs, files in os.walk(".", topdown=False):
		# for name in files:
		#     print(os.path.join(root, name))
		for name in dirs:
			if name.endswith('__pycache__'):
				dir_cache = os.path.join(root, name)
				# print('remove:', dir_cache)
				# os.remove(dir_cache)    #拒绝访问
				shutil.rmtree(dir_cache)

if __name__ == '__main__':
	pyProjClean()

	x = torch.rand(1,1,64,64)
	y = torch.rand(1,1,64,64)
	
	l1 = torch.norm(x-y, 2)#/64/64
	l2 = torch.nn.functional.mse_loss(x, y)
	l3 = torch.pow(x-y, 2).mean()
	print(l1.item(), l2.item(), l3.item())

	l1 = torch.norm(x)
	l2 = torch.pow(x, 2).mean()
	print(l1.item(), l2.item())

	l1 = torch.mean(torch.abs(x-y))
	l2 = torch.nn.functional.l1_loss(x, y)
	print(l1.item(), l2.item())



'''
Linux 删除文件夹和文件的命令
-r 就是向下递归，不管有多少级目录，一并删除
-f 就是直接强行删除，不作任何提示的意思

删除文件夹实例：
rm -rf /var/log/httpd/access
将会删除/var/log/httpd/access目录以及其下所有文件、文件夹

删除文件使用实例：
rm -f /var/log/httpd/access.log
将会强制删除/var/log/httpd/access.log这个文件
'''