import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

import warnings
from sklearn.metrics import *

plt.switch_backend('agg')

def adjust_learning_rate(optimizer, epoch, args):
	# lr = args.learning_rate * (0.2 ** (epoch // 2))
	if args.lradj == 'type1':
		lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
	elif args.lradj == 'type2':
		lr_adjust = {
			2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
			10: 5e-7, 15: 1e-7, 20: 5e-8
		}
	elif args.lradj == "cosine":
		lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
	if epoch in lr_adjust.keys():
		lr = lr_adjust[epoch]
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr
		print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
	def __init__(self, patience=7, verbose=False, delta=0):
		self.patience = patience
		self.verbose = verbose
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		self.val_loss_min = np.Inf
		self.delta = delta

	def __call__(self, val_loss, model, path):
		score = -val_loss
		if self.best_score is None:
			self.best_score = score
			self.save_checkpoint(val_loss, model, path)
		elif score < self.best_score + self.delta:
			self.counter += 1
			print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
			if self.counter >= self.patience:
				self.early_stop = True
		else:
			self.best_score = score
			self.save_checkpoint(val_loss, model, path)
			self.counter = 0

	def save_checkpoint(self, val_loss, model, path):
		
		if self.verbose:
			print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
   
		model_dict = model._get_trainable_stat()
  
		torch.save(model_dict, path + '/' + 'checkpoint.pth')
		self.val_loss_min = val_loss


class dotdict(dict):
	"""dot.notation access to dictionary attributes"""
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__


class StandardScaler():
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	def transform(self, data):
		return (data - self.mean) / self.std

	def inverse_transform(self, data):
		return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
	"""
	Results visualization
	"""
	plt.figure()
	plt.plot(true, label='GroundTruth', linewidth=2)
	if preds is not None:
		plt.plot(preds, label='Prediction', linewidth=2)
	plt.legend()
	plt.savefig(name, bbox_inches='tight')

def cal_metrics(y_pred, y_true):
	metrics = {}
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		metrics = classification_report(y_true, y_pred, output_dict=True)
	metrics['acc'] = np.mean(y_pred == y_true) 
	return metrics

def extract_useful_message(dataset, sequence, label_map, gt):
	if dataset == 'HAR':
		useful_msage = sequence.split("engaged in ")[-1]
		
	elif dataset == 'CT':
		useful_msage = sequence.split("pronouncing letter ")[-1]
		# print(useful_msage[0], self.label_map[int(samples['label'][idx])])
		# print(sequence)
		# print('==================================')
	
	elif dataset == 'FD':
		useful_msage = sequence.split("is currently looking at ")[-1]
  
	elif dataset == 'PD':
		useful_msage = sequence.split("is currently writing digit ")[-1]
  
	elif dataset == 'SAD':
		useful_msage = sequence.split("is currently pronouncing digit ")[-1]
		
		# print(sequence)
		# print('==================================')
		
	label_num = 0
	for idx, label_str in enumerate(label_map):
		if useful_msage.startswith(label_str):
		# if label_str in useful_msage:
			label_num = idx
			break

	# if label_map[label_num] != gt:
	# 	print(useful_msage)
	# 	print(label_map[label_num], gt)
	# 	print('==================================')

	return label_num

def plot_confusion_matrix(pred, true, name, n_class, save_dir):
	import seaborn as sns
	from sklearn.metrics import confusion_matrix # 导入计算混淆矩阵的包

	C1= confusion_matrix(true, pred) #True_label 真实标签 shape=(n,1);T_predict1 预测标签 shape=(n,1)
 
	xtick = ytick = [str(idx) for idx in range(1, n_class+1)]
 
	h = sns.heatmap(C1,annot=False,cbar=False) #画热力图,annot=True 代表 在图上显示 对应的值， fmt 属性 代表输出值的格式，cbar=False, 不显示 热力棒
	cb=h.figure.colorbar(h.collections[0]) #显示colorbar
	cb.ax.tick_params(labelsize=8) #设置colorbar刻度字体大小。
 
	plt.xticks(fontsize=8)
	plt.yticks(fontsize=8)
 
	# plt.show()
	plt.savefig(f'{save_dir}//confusion.png')

def test_params_flop(model,x_shape):
	"""
	If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
	"""
	model_params = 0
	for parameter in model.parameters():
		model_params += parameter.numel()
		print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
	import thop
	with torch.cuda.device(0):
		flops,params = thop.profile(model.cuda(),inputs=x_shape)
		flops, params = thop.clever_format([flops, params], '%.3f')
		print('flops:', flops)
		print('params:', params)