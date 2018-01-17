import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from resnet import resnet_mini



class Meta(nn.Module):


	def __init__(self):
		self.resnet = resnet_mini()
		repnet_sz = self.resnet(Variable(torch.rand(1, 3, 224, 224))).size()
		self.c = repnet_sz[1]
		self.d = repnet_sz[2]
		assert repnet_sz[2] == repnet_sz[3]
		print('repnet_sz:', repnet_sz)














	def forward(self, input, label):
		"""

		:param input: [batchsz, setsz, c, h, w]
		:param label: [batchsz, setsz]
		:return:
		"""
		batchsz, setsz, c_, h, w = input.size()
		x = self.resnet(input.view(batchsz * setsz, c_, h, w)).view(batchsz, setsz, )
		# [b, setsz, c_, h, w] => [b, 1, setsz, c_, h, w] => [b, setsz, setsz, c_, h, w]
		x_i = input.unsqueeze(1).expand(batchsz, setsz, setsz, c_, h, w)
		# [b, setsz, c_, h, w] => [b, setsz, 1, c_, h, w] => [b, setsz, setsz, c_, h, w]
		x_j = input.unsqueeze(2).expand(batchsz, setsz, setsz, c_, h, w)
		# [b, setsz, setsz, c_, h, w] => [b, setsz, setsz, 2c_]







if __name__ == '__main__':
	meta = Meta()

















