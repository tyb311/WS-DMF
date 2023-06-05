import torch, time
import torch.nn as nn


#start#
class DWCONV(nn.Module):
	def __init__(self, inp_c, out_channels, stride = 1):
		super(DWCONV, self).__init__()
		self.depthwise = nn.Conv2d(inp_c, out_channels, kernel_size = 3,
			stride = stride, padding = 1, groups = inp_c, bias = True
		)

	def forward(self, x):
		result = self.depthwise(x)
		return result

class LPU(nn.Module):
	def __init__(self, inp_c, out_channels):
		super(LPU, self).__init__()
		self.DWConv = DWCONV(inp_c, out_channels)

	def forward(self, x):
		result = self.DWConv(x) + x
		return result

class LMHSA(nn.Module):
	def __init__(self, channels, d_k, d_v, stride, heads):
		super(LMHSA, self).__init__()
		self.dwconv_k = DWCONV(channels, channels, stride=stride)
		self.dwconv_v = DWCONV(channels, channels, stride=stride)
		self.fc_q = nn.Linear(channels, heads * d_k)
		self.fc_k = nn.Linear(channels, heads * d_k)
		self.fc_v = nn.Linear(channels, heads * d_v)
		self.fc_o = nn.Linear(heads * d_k, channels)

		self.channels = channels
		self.d_k = d_k
		self.d_v = d_v
		self.stride = stride
		self.heads = heads
		self.scaled_factor = self.d_k ** -0.5
		self.num_patches = (self.d_k // self.stride) ** 2
		# self.B = nn.Conv2d(heads, heads, kernel_size=1, groups=heads)

	def forward(self, x):
		b, c, h, w = x.shape

		# Reshape
		x_reshape = x.view(b, c, h * w).permute(0, 2, 1)
		# x_reshape = nn.LayerNorm(c).cuda()(x_reshape)
		x_reshape = torch.nn.functional.layer_norm(x_reshape, (b, h * w, c))

		# Get q, k, v
		q = self.fc_q(x_reshape)
		q = q.view(b, h * w, self.heads, self.d_k).permute(0, 2, 1, 3).contiguous()  # [b, heads, h * w, d_k]

		k = self.dwconv_k(x)
		k_b, k_c, k_h, k_w = k.shape
		k = k.view(k_b, k_c, k_h * k_w).permute(0, 2, 1).contiguous()
		k = self.fc_k(k)
		k = k.view(k_b, k_h * k_w, self.heads, self.d_k).permute(0, 2, 1, 3).contiguous()  # [b, heads, k_h * k_w, d_k]

		v = self.dwconv_v(x)
		v_b, v_c, v_h, v_w = v.shape
		v = v.view(v_b, v_c, v_h * v_w).permute(0, 2, 1).contiguous()
		v = self.fc_v(v)
		v = v.view(v_b, v_h * v_w, self.heads, self.d_v).permute(0, 2, 1, 3).contiguous() # [b, heads, v_h * v_w, d_v]

		# Attention
		t1 = time.time()
		attn = torch.einsum('... i d, ... j d -> ... i j', q, k) * self.scaled_factor
		# attn = attn + self.B
		# attn = self.B(attn)
		# print(attn.shape)
		attn = torch.softmax(attn, dim = -1) # [b, heads, h * w, k_h * k_w]
		print('mhsa:', time.time() - t1)

		result = torch.matmul(attn, v).permute(0, 2, 1, 3)
		result = result.contiguous().view(b, h * w, self.heads * self.d_v)
		result = self.fc_o(result).view(b, self.channels, h, w)
		result = result + x
		return result

class CMTBlock(nn.Module):
	def __init__(self, inp_c=32, stride=1, d_k=32, d_v=32, num_heads=2, **args):
		super(CMTBlock, self).__init__()

		# Local Perception Unit
		self.lpu = LPU(inp_c, inp_c)

		# Lightweight MHSA
		self.lmhsa = LMHSA(inp_c, d_k, d_v, stride, num_heads)

		# Inverted Residual FFN
		self.irffn = nn.Sequential(
			DWCONV(inp_c, inp_c, 1),
			nn.BatchNorm2d(inp_c)
		)

	def forward(self, x):
		x = self.lpu(x)
		x = self.lmhsa(x)
		x = x + self.irffn(x)
		return x

class CMTResBlock(torch.nn.Module):
	def __init__(self, inp_c, out_c, ksize=3, shortcut=False, pool=True):
		super(CMTResBlock, self).__init__()
		pad = (ksize - 1) // 2

		block = []
		block.append(nn.Conv2d(inp_c, out_c, kernel_size=ksize, padding=pad))
		# block.append(nn.ReLU())
		# block.append(nn.BatchNorm2d(out_c))
		block.append(CMTBlock(out_c))

		if pool: self.pool = nn.MaxPool2d(kernel_size=2)
		else: self.pool = False

		self.block = nn.Sequential(*block)
	def forward(self, x):
		if self.pool: x = self.pool(x)
		out = self.block(x)
		return out
#end#


#	太费时间了，加个注意力投真难，所幸血管分割不需要长距离关系，local is enough
if __name__ == '__main__':
	net = CMTBlock()
	# net = CMTResBlock(32, 32)

	import time
	x = torch.rand(4, 32, 64, 64)
	st = time.time()
	y = net(x)
	print('time:', time.time()-st)
	print(y.shape)