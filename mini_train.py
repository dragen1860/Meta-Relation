import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import numpy as np
import os
from utils import make_imgs
from torch.optim import lr_scheduler
from rn import RN
from MiniImagenet import MiniImagenet
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from datetime import datetime


def eval(db, rn):
	"""
	for validation and test purpose.
	:param db:
	:param rn:
	:return:
	"""
	total_correct = 0
	total_num = 0
	total_val_loss = 0
	for j, batch in enumerate(db):
		# batch : ([10,10,3,84,84], [10,10], [10,75,3,84,84], [10,75])
		support_x = Variable(batch[0]).cuda()
		support_y = Variable(batch[1]).cuda()
		query_x = Variable(batch[2]).cuda()
		query_y = Variable(batch[3]).cuda()

		rn.eval()
		pred, correct, val_loss = rn(support_x, support_y, query_x, query_y, False)

		total_val_loss += val_loss.data[0]
		total_correct += correct.cpu().data[0]
		total_num += query_x.size(0) * query_x.size(1)
	return total_correct / total_num, total_val_loss


if __name__ == '__main__':
	resize = 184
	n_way = 5
	k_shot = 5
	n_query_per_cls = 1
	batchsz = 9
	rn = RN(resize, 3).cuda()
	mdl_file = 'ckpt/rn55_%d%d.mdl'%(rn.c, rn.d)

	if os.path.exists(mdl_file):
		print('load checkpoint ...', mdl_file)
		rn.load_state_dict(torch.load(mdl_file))

	model_parameters = filter(lambda p: p.requires_grad, rn.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('total params:', params)

	optimizer = optim.Adam(rn.parameters(), lr=1e-5, weight_decay=1e-4)
	scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose = True)
	tb = SummaryWriter('runs', str(datetime.now()))

	best_accuracy = 0
	for epoch in range(1000):
		mini = MiniImagenet('../mini-imagenet/', mode='train', n_way=n_way, k_shot=k_shot, k_query=1, batchsz=10000, resize=resize)
		db = DataLoader(mini,  batchsz, shuffle=True, num_workers=6)
		mini_val = MiniImagenet('../mini-imagenet/',  mode='val', n_way=n_way, k_shot=k_shot, k_query=1, batchsz=100, resize=resize)
		db_val = DataLoader(mini_val, batchsz, shuffle=True)


		for step, batch in enumerate(db):
			# batch : ([10,10,3,84,84], [10,10], [10,75,3,84,84], [10,75])
			support_x = Variable(batch[0]).cuda()
			support_y = Variable(batch[1]).cuda()
			query_x = Variable(batch[2]).cuda()
			query_y = Variable(batch[3]).cuda()

			rn.train()
			loss, score = rn(support_x, support_y, query_x, query_y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if step % 50 == 0 :
				accuracy, total_val_loss = eval(db_val, rn)

				tb.add_scalar('accuracy', accuracy)
				print('accuracy:', accuracy, 'best accuracy:', best_accuracy, 'val loss:', total_val_loss)
				# update learning rate per epoch
				scheduler.step(total_val_loss)


				if accuracy > best_accuracy:
					best_accuracy = accuracy
					torch.save(rn.state_dict(), mdl_file)
					print('saved to checkpoint:',mdl_file)

					print('now conduct test performance...')
					mini_test = MiniImagenet('../mini-imagenet/', mode='test', n_way=n_way, k_shot=k_shot, k_query=1,
					                        batchsz=200, resize=resize)
					db_test = DataLoader(mini_test, batchsz, shuffle=True)
					accuracy_test, _ = eval(db_test, rn)
					print('>>>>>>>>>>>> test accuracy:', accuracy_test, '<<<<<<<<<<<<<<')



			if step% 15 == 0 and step != 0:
				tb.add_scalar('loss', loss.cpu().data[0])
				print('%d-way %d-shot %d batch> epoch:%d step:%d, loss:%f'%(n_way, k_shot, batchsz, epoch, step, loss.cpu().data[0]))