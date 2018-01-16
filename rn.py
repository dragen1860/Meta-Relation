import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch import optim
from torchvision.models import resnet18, resnet34, resnet50
import numpy as np



class RN(nn.Module):
	"""
	Relational Network for few-shot Learning
	RN includes two parts of network, the first part is feature representation, which use traditional CNN to learn feature
	representation, and the second part is relational part, which use combination of two features from previous net to explicityly
	compare the same-class or diff-class relation.
	for same-class combination, the network should output 1 and 0 for diff-class.
	In one-shot setting, there is only one same-class pairs, therefore the target = argmax(output)
	"""
	def __init__(self, resize, kernelsz):
		super(RN, self).__init__()

		self.build(resize, kernelsz)

	def build(self, resize, kernelsz = 3):
		"""
		build forward network structure
		:return:
		"""
		# build feature representation net
		# build feature representation net
		resnet = resnet18(pretrained=True)
		modules = list(resnet.children())[:-2]
		self.repnet = nn.Sequential(*modules,
		                            nn.Conv2d(512,256, kernel_size = kernelsz),
		                            nn.ReLU(inplace=True)) # [c, d, d]
		repnet_sz = self.repnet(Variable(torch.randn((1, 3, resize, resize)))).size()
		self.d = repnet_sz[2] # this is the size of repnet
		self.c = repnet_sz[1] # this is the channel of repnet, not general channel
		print('repnet size:', repnet_sz) # (1,64,15,15)

		# we are uncertain the dim of features here
		# therefore the input dim of relation net is 512*2
		self.g = nn.Sequential(nn.Linear( (self.c + 2) * 2, 128),
		                       nn.ReLU(inplace=True),
		                       nn.Linear(128, 64),
		                       nn.ReLU(inplace=True))

		self.f = nn.Sequential(nn.Linear(64, 16),
		                       nn.ReLU(inplace=True),
		                       nn.Linear(16, 1),
		                       nn.Sigmoid())

		coord = np.array([(i/self.d , j/self.d) for i in range(self.d) for j in range(self.d)])
		self.coord = torch.from_numpy(coord).float().view(self.d, self.d, 2).transpose(0, 2).transpose(1,2).contiguous()
		self.coord = self.coord.unsqueeze(0).unsqueeze(0)
		print('self.coord:', self.coord.size()) # [2, self.d, self.d]

	def pretrain(self):
		pass

	def forward(self, input, input_y, query, query_y, train = True):
		"""
		n_way vs setsz
		n_query vs samples_per_class

		:param input: [batch, setsz, c, h, w]
		:param query: [batch, querysz, c, h, w]
		:param input_y: [batch, setsz]
		:param query_y: [batch, querysz]
		:return:
		"""
		batchsz, setsz, c, h, w = input.size()
		querysz = query.size(1) # querysz = n_way * n_query_per_cls
		d = self.d # dim of relation net input size, get by forward repnet in __init__

		# as traditional conv net only support 4d input, we need to flatten first 2d data
		input_f = input.view(batchsz * setsz, c, h, w)
		query_f = query.view(batchsz * querysz, c, h, w)
		# forward representation network to retain features
		# the output size of repnet is : [batch, self.c, d, d]
		input_f = self.repnet(input_f).view(batchsz, setsz, self.c, d, d) # c: raw image channel, self.c: repnet channel output
		query_f = self.repnet(query_f).view(batchsz, querysz, self.c, d, d)

		## now make the combination between two pairs
		# include the coordinate information in each feature dim
		input_f = torch.cat([input_f, Variable(self.coord.expand(batchsz, setsz, 2, d, d)).cuda()], dim = 2)
		query_f = torch.cat([query_f, Variable(self.coord.expand(batchsz, querysz, 2, d, d)).cuda()], dim = 2)
		self.c += 2
		# [b, setsz, self.c, d*d] => [b, 1, setsz, self.c, d*d] => [b, 1, setsz, self.c, 1, d*d]
		input_f = input_f.view(batchsz, setsz, self.c, d*d).unsqueeze(1).unsqueeze(4).expand(batchsz, querysz, setsz, self.c, d*d, d*d)
		# [b, querysz, 1, self.c, d*d] => [b, querysz, 1, self.c, d*d, 1] => [b, querysz, setsz, self.c, d*d, d*d]
		query_f = query_f.view(batchsz, querysz, self.c, d*d).unsqueeze(2).unsqueeze(5).expand(batchsz, querysz, setsz, self.c, d*d, d*d)
		# [b, querysz, setsz, self.c * 2, d*d, d*d]
		comb = torch.cat([input_f, query_f], dim=3)

		# [b, querysz, setsz, self.c * 2, d*d, d*d] => [b, querysz, setsz, d*d, d*d, self.c * 2]
		x_full = comb.transpose(3, 5).contiguous().view(batchsz * querysz * setsz * d*d * d*d, self.c * 2)
		# push to g network
		x_ = self.g(x_full)
		# sum over coordinate axis and squeeze it
		x_g = x_.view(batchsz * querysz * setsz, d*d * d*d, -1) # the last dim can be derived by layer setting
		x_g = x_g.sum(1) # [batchsz * querysz * setsz, -1]
		# push to f network
		x_f = self.f(x_g) # [batchsz * querysz * setsz, 1]
		score = x_f.view(batchsz, querysz, setsz, 1) # [batch, querysz, setsz, 1]

		## update self.c for next forward procedure
		self.c -= 2

		# now build its label
		# input_y: [batchsz, setsz]
		# query_y: [batchsz, querysz]
		# [b, setsz] => [b, setsz, 1] => [b, 1, setsz, 1] => [b, querysz, setsz, 1]
		input_y_ = input_y.unsqueeze(2).unsqueeze(1).expand(batchsz, querysz, setsz, 1)
		# [b, querysz] => [b, querysz, 1] => [b, querysz, 1, 1] => [b, querysz, setsz, 1]
		query_y_ = query_y.unsqueeze(2).unsqueeze(2).expand(batchsz, querysz, setsz, 1)
		label_pair = torch.cat([input_y_, query_y_], dim=3)  # [b, n_query, setsz, 2]
		label0 = label_pair[..., 0]  # [b, n_query, setsz]
		label1 = label_pair[..., 1]  # [b, n_query, setsz]
		# convert bytetensor to floattensor
		# NOT EQUAL set 1, equal set 0
		label = torch.eq(label0, label1).unsqueeze(3).float()  # [b, querysz, setsz] => [b, querysz, setsz, 1]


		if train:
			# score: [batch, querysz, setsz, 1]
			# label: [b, querysz, setsz, 1]
			# TODO: add cross-entropy rn loss and classification loss
			loss = torch.pow(label - score, 2).sum()
			# print(label, score)
			#
			return loss, score
		else: # just predict
			# score: [batch, querysz, setsz, 1]
			# label: [b, querysz, setsz, 1]
			val_loss = torch.pow(label - score, 2).sum()

			# [batch, n_query, setsz]
			_, idx = score.abs().squeeze(3).max(2) # [b, querysz]
			# input_y: [batchsz, setsz]
			# pred: [batchsz, querysz]
			pred = torch.gather(input_y, dim=1, index=idx)
			correct = torch.eq(pred, query_y).sum()
			return pred, correct, val_loss
























