import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
import csv
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Solver(object):
	def __init__(self, config, train_loader, valid_loader, test_loader):

		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader

		# Models
		self.unet = None
		self.optimizer = None
		self.scheduler = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		self.criterion = torch.nn.MSELoss()
		self.augmentation_prob = config.augmentation_prob

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2

		# Training settings
		self.num_epochs = config.num_epochs
		self.num_epochs_decay = config.num_epochs_decay
		self.batch_size = config.batch_size

		# Step size
		self.log_step = config.log_step
		self.val_step = config.val_step

		# Path
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.mode = config.mode

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_type = config.model_type
		self.t = config.t
		self.build_model()

	def build_model(self):
		"""Build generator and discriminator."""
		if self.model_type =='U_Net':
			self.unet = U_Net(img_ch=1,output_ch=1)
		elif self.model_type =='R2U_Net':
			self.unet = R2U_Net(img_ch=1,output_ch=1,t=self.t)
		elif self.model_type =='AttU_Net':
			self.unet = AttU_Net(img_ch=1,output_ch=1)
		elif self.model_type == 'R2AttU_Net':
			self.unet = R2AttU_Net(img_ch=1,output_ch=1,t=self.t)
			

		self.optimizer = optim.Adam(list(self.unet.parameters()),
									  self.lr, (self.beta1, self.beta2)) # weight_decay=0.001
		# self.optimizer = optim.SGD(list(self.unet.parameters()),
		# 							self.lr, momentum=0.99, nesterov=True)
		self.unet.to(self.device)
		self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)
		# self.print_network(self.unet, self.model_type)

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))

	def to_data(self, x):
		"""Convert variable to tensor."""
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data

	def update_lr(self, g_lr, d_lr):
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr

	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.unet.zero_grad()

	def compute_accuracy(self,SR,GT):
		SR_flat = SR.view(-1)
		GT_flat = GT.view(-1)

		acc = GT_flat.data.cpu()==(SR_flat.data.cpu()>0.5)

	def tensor2img(self,x):
		img = (x[:,0,:,:]>x[:,1,:,:]).float()
		img = img*255
		return img


	def train(self):
		"""Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		#===========================================================================================#
		
		unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))
		# unet_path = ''
		start_epoch = -1
		# U-Net Train
		if os.path.isfile(unet_path):
			# Load the pretrained Encoder
			checkpoint = torch.load(unet_path)  # 加载断点
			self.unet.load_state_dict(checkpoint)
			print('%s is Successfully Loaded from %s'%(self.model_type,unet_path))
			# start_epoch = checkpoint['epoch']  # 设置开始的epoch
			# self.optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
			# self.scheduler.load_state_dict(checkpoint['lr_schedule'])  # 加载lr_scheduler

		# Train for Encoder
		# lr = self.lr
		best_unet_score = 0.
		print('best_unet_score: ', best_unet_score)
		print('optimizer: ', self.optimizer)
		print('learning rate:',self.optimizer.state_dict()['param_groups'][0]['lr'])

		for epoch in range(start_epoch+1, self.num_epochs):
			print('Epoch: ', epoch)

			self.unet.train(True)
			epoch_loss = 0

			acc = 0.	# Accuracy
			length = 0

			for i, (images, GT) in enumerate(self.train_loader):
				# GT : Ground Truth

				images = images.to(self.device)
				GT = GT.to(self.device)
				images = images.type(torch.cuda.FloatTensor)
				GT = GT.type(torch.cuda.FloatTensor)
				# SR : Segmentation Result
				SR = self.unet(images)

				loss = self.criterion(SR,GT)
				epoch_loss += loss.item()
				# print(epoch_loss)

				# Backprop + optimize
				self.reset_grad()
				loss.backward()
				self.optimizer.step()


				# acc += get_accuracy(SR,GT)

				# length += images.size(0)
				length += 1

			self.scheduler.step()
			acc = acc/length
			epoch_loss = epoch_loss/length

			# Print the log info
			print('Epoch [%d/%d], Loss: %.8f, \n[Training] Acc: %.8f' % (
				  epoch+1, self.num_epochs, epoch_loss, acc))

			result2txt = 'iteration:' + str(epoch) + '\ntrain_loss:' + str(
				'{:.8f}'.format(epoch_loss)) + '\ntrain_fidelity:' + \
						 str('{:.8f}'.format(acc))
			file_name = '4loss_attunet'
			with open(file_name + '.txt', 'a') as file_handle:  # .txt可以不自己新建,代码会自动新建
				file_handle.write(result2txt)  # 写入
				file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据

			# Decay learning rate
			# if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
			# 	lr -= (self.lr / float(self.num_epochs_decay))
			# 	for param_group in self.optimizer.param_groups:
			# 		param_group['lr'] = lr
			# 	print ('Decay learning rate to lr: {}.'.format(lr))

			print ('Decay learning rate to lr: {}.'.format(self.scheduler.get_last_lr()))


			#===================================== Validation ====================================#
			self.unet.train(False)
			self.unet.eval()

			epoch_loss_val = 0.
			acc = 0.	# Accuracy
			length=0
			for i, (images, GT) in enumerate(self.valid_loader):
				images = images.to(self.device)
				GT = GT.to(self.device)
				images = images.type(torch.cuda.FloatTensor)
				GT = GT.type(torch.cuda.FloatTensor)
				SR = self.unet(images)

				loss = self.criterion(SR, GT)
				epoch_loss_val += loss.item()
				acc += get_accuracy(SR,GT)


				# length += images.size(0)
				length += 1

			epoch_loss_val =epoch_loss_val/length
			acc = acc/length
			unet_score = acc



			print('Epoch [%d/%d], Loss: %.8f, \n[Validation] Acc: %.8f' % (
				epoch + 1, self.num_epochs, epoch_loss_val, acc))
			# print('[Validation] Acc: %.4f'%(acc))

			'''
			torchvision.utils.save_image(images.data.cpu(),
										os.path.join(self.result_path,
													'%s_valid_%d_image.png'%(self.model_type,epoch+1)))
			torchvision.utils.save_image(SR.data.cpu(),
										os.path.join(self.result_path,
													'%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
			torchvision.utils.save_image(GT.data.cpu(),
										os.path.join(self.result_path,
													'%s_valid_%d_GT.png'%(self.model_type,epoch+1)))
			'''

			result2txt = 'iteration:' + str(epoch) + '\nval_loss:' + str(
				'{:.8f}'.format(epoch_loss_val)) + '\nval_fidelity:' + \
						 str('{:.8f}'.format(acc))

			with open(file_name + '.txt', 'a') as file_handle:  # .txt可以不自己新建,代码会自动新建
				file_handle.write(result2txt)  # 写入
				file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据

			# Save Best U-Net model
			if unet_score > best_unet_score:
				best_unet_score = unet_score
				best_epoch = epoch
				best_unet = self.unet.state_dict()
				print('Best %s model score : %.4f'%(self.model_type,best_unet_score))
				torch.save(best_unet,unet_path)
					
			#===================================== Test ====================================#
	def test(self):
			del self.unet
			# del best_unet
			self.build_model()
			self.unet.load_state_dict(torch.load('models/AttU_Net-200-0.0010-100-0.0000.pkl'))
			
			self.unet.train(False)
			self.unet.eval()

			acc = 0.	# Accuracy

			length=0
			start = time.time()
			for i, (images, GT) in enumerate(self.test_loader):
				images = images.to(self.device)
				GT = GT.to(self.device)
				images = images.type(torch.cuda.FloatTensor)
				GT = GT.type(torch.cuda.FloatTensor)
				SR = self.unet(images)

				acc += get_accuracy(SR,GT)
						
				length += 1
			end = time.time()
			print('time is ', end-start)
			acc = acc/length

			print('fidelity is ', np.real(acc))
			f = open(os.path.join(self.result_path,'result.csv'), 'a', encoding='utf-8', newline='')
			wr = csv.writer(f)
			wr.writerow([self.model_type,acc,self.lr,self.num_epochs,self.num_epochs_decay,self.augmentation_prob])
			f.close()
			

			
