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
from meta import Meta

def eval(db, meta):
	"""
	for validation and test purpose.
	:param db:
	:param rn:
	:return:
	"""
	total_correct = 0
	total_num = 0
	total_loss = 0
	display_onebatch = False
	for i, batch in enumerate(db):
		support_x = Variable(batch[0]).cuda()
		support_y = Variable(batch[1]).cuda()
		query_x = Variable(batch[2]).cuda()
		query_y = Variable(batch[3]).cuda()

		meta.eval()
		correct, total, pred = meta.predict(support_x, support_y, query_x, query_y)

		total_correct += correct
		total_num += total

		if not display_onebatch:
			display_onebatch = True  # only display once
			all_img, max_width = make_imgs(n_way, k_shot, k_query, support_x.size(0),
			                               support_x, support_y, query_x, query_y, pred)
			all_img = make_grid(all_img, nrow=max_width)
			tb.add_image('result batch', all_img)

	return total_correct / total_num


if __name__ == '__main__':
	resize = 224
	n_way = 5
	k_shot = 1
	k_query = 1
	batchsz = 3
	meta = Meta(n_way, k_shot).cuda()
	mdl_file = 'ckpt/meta%d%d.mdl' % (n_way, k_shot)
	pretrain_mdl_file = 'ckpt/pretrain.mdl'

	# if os.path.exists(mdl_file):
	# 	print('>>load checkpoint ...', mdl_file)
	# 	meta.load_state_dict(torch.load(mdl_file))
	# elif os.path.exists(pretrain_mdl_file):
	# 	print('>>load pretrain ...', pretrain_mdl_file)
	# 	meta.load_state_dict(torch.load(pretrain_mdl_file))

	model_parameters = filter(lambda p: p.requires_grad, meta.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('total params:', params)

	optimizer = optim.Adam(meta.parameters(), lr=1e-5, weight_decay=1e-4)
	scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
	tb = SummaryWriter('runs', str(datetime.now()))

	best_val_acc = 0
	for epoch in range(1000):
		mini = MiniImagenet('../mini-imagenet/', mode='train', n_way=n_way, k_shot=k_shot, k_query=k_query, batchsz=1000,
		                    resize=resize)
		db = DataLoader(mini, batchsz, shuffle=True, num_workers=6)
		mini_val = MiniImagenet('../mini-imagenet/', mode='val', n_way=n_way, k_shot=k_shot, k_query=k_query, batchsz=100,
		                        resize=resize)
		db_val = DataLoader(mini_val, batchsz, shuffle=True)

		for step, batch in enumerate(db):
			# batch : ([10,10,3,84,84], [10,10], [10,75,3,84,84], [10,75])
			support_x = Variable(batch[0]).cuda()
			support_y = Variable(batch[1]).cuda()
			query_x = Variable(batch[2]).cuda()
			query_y = Variable(batch[3]).cuda()

			meta.train()
			cls_loss, euc_loss, sym_score = meta(support_x, support_y, query_x, query_y)
			optimizer.zero_grad()
			euc_loss.backward()
			optimizer.step()


			if step % 100 == 0:
				val_acc = eval(db_val, meta)

				tb.add_scalar('accuracy', val_acc)
				print('accuracy:', val_acc, 'best accuracy:', best_val_acc)
				# update learning rate per epoch
				# scheduler.step(total_val_loss)

				if val_acc > best_val_acc:
					best_val_acc = val_acc
					torch.save(meta.state_dict(), mdl_file)
					print('saved to checkpoint:', mdl_file)

					if val_acc > 0.4:
						print('now conduct test performance...')
						mini_test = MiniImagenet('../mini-imagenet/', mode='test', n_way=n_way, k_shot=k_shot,
						                         k_query=1,
						                         batchsz=200, resize=resize)
						db_test = DataLoader(mini_test, batchsz, shuffle=True)
						test_acc, _ = eval(db_test, meta)
						print('>>>>>>>>>>>> test accuracy:', test_acc, '<<<<<<<<<<<<<<')


			if step % 15 == 0 and step != 0:
				print('%d-way %d-shot %d batch> epoch:%d step:%d, cls loss:%f, euc loss:%f, sym scores:%f' % (
				n_way, k_shot, batchsz, epoch, step, cls_loss.cpu().data[0], euc_loss.cpu().data[0], sym_score.cpu().data[0]))
