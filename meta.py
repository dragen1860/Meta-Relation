import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from resnet import resnet_mini


class Meta(nn.Module):
	def __init__(self):
		super(Meta, self).__init__()

		self.resnet = resnet_mini()
		repnet_sz = self.resnet(Variable(torch.rand(1, 3, 224, 224)))[0].size()
		self.c = repnet_sz[1]
		self.d = repnet_sz[2]
		assert repnet_sz[2] == repnet_sz[3]
		print('repnet_sz:', repnet_sz)

		coord = np.array([(i / self.d, j / self.d) for i in range(self.d) for j in range(self.d)])
		self.coord = torch.from_numpy(coord).float().view(self.d, self.d, 2).transpose(0, 2).transpose(1, 2).contiguous()
		# [2, d, d] => [b, setsz, 2, d, d]
		self.coord = self.coord.unsqueeze(0).unsqueeze(0)
		print('self.coord:', self.coord.size())  # [1, 1, 2, self.d, self.d]

		# relational module
		self.g = nn.Sequential(nn.Linear((self.c + 2) * 2, 128),
		                       nn.ReLU(inplace=True),
		                       nn.Linear(128, 64),
		                       nn.ReLU(inplace=True))

		self.f = nn.Sequential(nn.Linear(64, 16),
		                       nn.ReLU(inplace=True),
		                       nn.Linear(16, 1),
		                       nn.Sigmoid())

		self.criteon = nn.CrossEntropyLoss()



	def pretrain(self, support_x, support_y, query_x, query_y):
		"""
		in pre-training stage, we will merage all support_x and query_x data and then compose a large relational matrix.
		This function will only train classifier network, not relation network.
		After the pre-training, it will save a chkpoint for classification network.
		:param support_x: [b, setsz, c_, h, w]
		:param support_y: [b, setsz]
		:param query_x:   [b, querysz, c_, h, w]
		:param query_y:   [b, querysz]
		:return:
		"""
		# [b, setsz, c_, h, w]+ [b, querysz, c_, h, w] => [batchsz, setsz+querysz, c_, h, w]
		input = torch.cat([support_x, query_x], dim = 1)
		# [b, setsz] + [b, querysz] => [b, setsz+querysz]
		label = torch.cat([support_y, query_y], dim = 1)
		# from here on, we update setsz = setsz + querysz
		batchsz, setsz, c_, h, w = input.size()
		c, d = self.c, self.d

		## Combination bewteen two images, between objects in two images
		# get feature from [b, setsz, c_, h, w] => [b*setsz, c, d, d]
		# cls_pred/logits is the prediction of classification, => [batchsz*setsz, 64]
		# the output logits doesn't have softmax
		x, logits = self.resnet(input.view(batchsz * setsz, c_, h, w))

		# cls_pred: [b*setsz, 64]
		# label: [b, setsz] => [b*setsz]
		loss = self.criteon(logits, label.view(-1))

		# calculate train accuracy
		# [b*setsz, 64] => [b*setsz, 64]
		prob = F.softmax(logits, dim = 1)
		# [b*setsz, 64] => [b*setsz]
		_, indices = torch.max(prob, dim = 1)
		# => [b, setsz]
		indices = indices.view(batchsz, setsz)
		# label: [b, setsz]
		correct = torch.eq(indices, label).sum().data[0]
		total = batchsz * setsz


		return loss, correct/total

	def forward(self, support_x, support_y, query_x, query_y):
		"""
		in training stage, we will merage all support_x and query_x data and then compose a large relational matrix.
		The matrix includes the similar relation between each labels and the pred matrix should converge to target matrix.
		besides, the matrix should keep symmetric.
		:param support_x: [b, setsz, c_, h, w]
		:param support_y: [b, setsz]
		:param query_x:   [b, querysz, c_, h, w]
		:param query_y:   [b, querysz]
		:return:
		"""
		# [b, setsz, c_, h, w]+ [b, querysz, c_, h, w] => [batchsz, setsz+querysz, c_, h, w]
		input = torch.cat([support_x, query_x], dim = 1)
		# [b, setsz] + [b, querysz] => [b, setsz+querysz]
		label = torch.cat([support_y, query_y], dim = 1)
		# from here on, we update setsz = setsz + querysz
		batchsz, setsz, c_, h, w = input.size()
		c, d = self.c, self.d

		## Combination bewteen two images, between objects in two images
		# get feature from [b, setsz, c_, h, w] => [b*setsz, c, d, d]
		# cls_pred is the prediction of classification, => [batchsz*setsz, 64]
		x, cls_pred = self.resnet(input.view(batchsz * setsz, c_, h, w))
		# [b*setsz, c, d, d] => [b, setsz, c, d, d] => [b, setsz, c+2, d, d]
		x = torch.cat([x.view(batchsz, setsz, c, d, d), Variable(self.coord.expand(batchsz, setsz, 2, d, d)).cuda()], dim=2)
		# update c, DO not update self.c
		c += 2
		# [b, setsz, c, d, d] => [b, setsz, c, d*d]
		x = x.view(batchsz, setsz, c, d * d)

		# [b, setsz, c, d*d] => [b, 1, setsz, c, d*d] => [b, 1, setsz, c, 1, d*d] => [b, setsz, setsz, c, d*d, d*d]
		x_i = x.unsqueeze(1).unsqueeze(4).expand(batchsz, setsz, setsz, c, d*d, d*d)
		# [b, setsz, c, d*d] => [b, setsz, 1, c, d*d, 1] => [b, setsz, setsz, c, d*d, d*d]
		x_j = x.unsqueeze(2).unsqueeze(5).expand(batchsz, setsz, setsz, c, d*d, d*d)
		# [b, setsz, setsz, 2c, d*d, d*d]
		comb = torch.cat([x_i, x_j], dim=3)

		# [b, setsz, setsz, 2c, d*d, d*d] => [b, setsz, setsz, d*d, d*d, 2c] => [b*setsz*setsz*d^4, 2c]
		x_rn = comb.transpose(3, 5).contiguous().view(batchsz * setsz * setsz * d*d * d*d, c * 2)
		# push to G network
		# [b*setsz*setsz*d^4, 2c] => [b*setsz*setsz*d^4, new_dim]
		x_f = self.g(x_rn)
		# sum over coordinate axis, erase spatial dims => [batchsz * setsz * setsz, -1]
		x_f = x_f.view(batchsz * setsz * setsz, d * d * d * d, -1).sum(1)  # the last dim can be derived by layer setting
		# push to F network
		# [batchsz * setsz * setsz, -1] => [batchsz * setsz * setsz, new_dim] => [batch, setsz, setsz, 1]
		pred = self.f(x_f).view(batchsz, setsz, setsz, 1)


		## now build its label
		# [b, setsz] => [b, setsz, 1] => [b, 1, setsz, 1] => [b, setsz, setsz, 1]
		label_i = label.unsqueeze(2).unsqueeze(1).expand(batchsz, setsz, setsz, 1)
		# [b, setsz] => [b, setsz, 1] => [b, setsz, 1, 1] => [b, setsz, setsz, 1]
		label_j = label.unsqueeze(2).unsqueeze(2).expand(batchsz, setsz, setsz, 1)
		label_pair = torch.cat([label_i, label_j], dim=3)  # [b, n_query, setsz, 2]
		label0 = label_pair[..., 0]  # [b, setsz, setsz]
		label1 = label_pair[..., 1]  # [b, setsz, setsz]
		# convert bytetensor to floattensor
		y = torch.eq(label0, label1).unsqueeze(3).float()  # [b, setsz, setsz] => [b, setsz, setsz, 1]

		## calculate loss
		# 1. calcuate symmetric loss, pls refer here: https://math.stackexchange.com/questions/2048817/metric-for-how-symmetric-a-matrix-is
		# pred: [b, setsz, setsz, 1]
		# sym: [b, setsz, setsz]
		# anti:[b, setsz, setsz]
		sym = 0.5 * (pred.squeeze(3) + pred.squeeze(3).transpose(1, 2))
		anti = 0.5 * (pred.squeeze(3) - pred.squeeze(3).transpose(1, 2))
		# sym_loss is [-1, 1], 1 stands for symtric while -1 stands for anti-symmetric
		# sym_norm: [b, setsz, setz] => [b, setsz] => [b]
		sym_norm = torch.norm(sym, 2, -1).norm(2, -1)
		# anti_norm: [b, setsz, setz] => [b, setsz] => [b]
		anti_norm = torch.norm(anti, 2, -1).norm(2, -1)
		# [b] - [b] and divide by [b] + [b] => [b]
		sym_loss = ((sym_norm - anti_norm) / (sym_norm + anti_norm)).mean()
		# euclidean distance
		euc_loss = torch.pow(pred - y, 2).mean()
		# classificatio loss, cls_pred: [b * setsz, 64], label: [b, setsz] => [b*setsz]
		cls_loss = self.criteon(cls_pred, label.view(-1))
		# sum over loss
		rn_loss = euc_loss - 0.1 * sym_loss

		return cls_loss, rn_loss, sym_loss



if __name__ == '__main__':
	import numpy as np
	from torch.optim import lr_scheduler
	from MiniImagenet import MiniImagenet
	from torch.utils.data import DataLoader

	resize = 224
	n_way = 5
	k_shot = 5
	k_query = 1
	batchsz = 4
	rn = Meta().cuda()
	mdl_file = 'ckpt/pretain.mdl'


	model_parameters = filter(lambda p: p.requires_grad, rn.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('total params:', params)

	optimizer = optim.Adam(rn.parameters(), lr=1e-5, weight_decay=1e-5)
	scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

	best_accuracy = 0
	for epoch in range(1000):
		mini = MiniImagenet('../mini-imagenet/', mode='train', n_way=n_way, k_shot=k_shot, k_query=k_query,
		                    batchsz=10000, resize=resize)
		db = DataLoader(mini, batchsz, shuffle=True, num_workers=6)

		for step, batch in enumerate(db):
			# batch : ([10,10,3,84,84], [10,10], [10,75,3,84,84], [10,75])
			support_x = Variable(batch[0]).cuda()
			support_y = Variable(batch[1]).cuda()
			query_x = Variable(batch[2]).cuda()
			query_y = Variable(batch[3]).cuda()


			rn.train()
			cls_loss, train_acc = rn.pretrain(support_x, support_y, query_x, query_y)
			optimizer.zero_grad()
			cls_loss.backward()
			optimizer.step()

			if step % 15 == 0 and step != 0:
				print('%d-way %d-shot %d batch> epoch:%d step:%d, cls loss:%f, train acc:%f' % (
				n_way, k_shot, batchsz, epoch, step, cls_loss.cpu().data[0], train_acc))


			# rn.train()
			# cls_loss, rn_loss, sym_loss = rn(support_x, support_y, query_x, query_y)
			# optimizer.zero_grad()
			# cls_loss.backward(retain_graph = True)
			# rn_loss.backward()
			# optimizer.step()
			#
			# if step % 15 == 0 and step != 0:
			# 	print('%d-way %d-shot %d batch> epoch:%d step:%d, cls loss:%f, rn loss:%f, sym loss:%f' % (
			# 	n_way, k_shot, batchsz, epoch, step, cls_loss.cpu().data[0], rn_loss.cpu().data[0], sym_loss.cpu().data[0]))