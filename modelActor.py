from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import random

from dataloader import *

import tensorflow as tf

import model

from utils import load, dump

from client import Client

import os
import sys

try:
	import torchvision.transforms as transforms
	import torch
	def seed_torch(seed=615):
	    random.seed(seed)
	    np.random.seed(seed)
	    torch.manual_seed(seed)
	    torch.cuda.manual_seed(seed)
	    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	    torch.backends.cudnn.benchmark = False
	    torch.backends.cudnn.deterministic = True

	seed_torch()
except:
	pass


class ModelActor(object):
	def __init__(self, FLAGS, worker_id, tr_client_ids, val_client_ids, te_client_ids, num_workers, use_gpu):
		self.worker_id = worker_id
		self.client_ids = np.concatenate((tr_client_ids, val_client_ids, te_client_ids))
		self.tr_client_ids = tr_client_ids
		self.val_te_client_ids = np.concatenate((val_client_ids, te_client_ids))

		self.num_workers = num_workers
		self.FLAGS = FLAGS
		self.use_gpu = use_gpu
		self.model = None
		self.client_data = {}

		self.batch_size = FLAGS.batch_size
		
		self.learning_rate = self.FLAGS.learning_rate

		self.save_fn = self.FLAGS.out_path+'/worker_{}.ckpt'.format(worker_id)
		self.save_fn_best = self.FLAGS.out_path+'/worker_{}_best.ckpt'.format(worker_id)

		if FLAGS.modelname == 'mbnt':
			self.tr_transforming = transforms.Compose([
				transforms.ToPILImage(),
				transforms.Resize(224),
				transforms.Pad(28),
				transforms.RandomCrop(224),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])
			self.val_transforming = transforms.Compose([
				transforms.ToPILImage(),
				transforms.Resize(224),
				transforms.ToTensor(),
				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])

	def init_graph(self):
		with tf.device('/gpu:0' if self.use_gpu else '/cpu:0'):
			print('initializing graph')
			if self.FLAGS.modelname == 'FFN':
				self.model = model.FFN(
					learning_rate=self.FLAGS.learning_rate, 
					feature_size=self.FLAGS.feature_size, 
					num_classes=self.FLAGS.num_classes,
					hidden_size=self.FLAGS.hidden_size)
			elif self.FLAGS.modelname == 'mbnt':
				self.model = model.MobilenetV2(
					learning_rate=self.FLAGS.learning_rate,
					num_classes=self.FLAGS.num_classes, 
					input_size=224, 
					dropout_keep_prob=0.8)
			elif self.FLAGS.modelname == 'LR':
				self.model = model.LogisticRegression(
					learning_rate=self.FLAGS.learning_rate,
					feature_size=self.FLAGS.feature_size,
					num_classes=self.FLAGS.num_classes)
			elif self.FLAGS.modelname == 'lstm':
				self.model = model.StackedLSTM(seq_len=80, 
					num_classes=self.FLAGS.num_classes, 
					n_hidden=256)

			self.model.build_graph(optimizer=self.FLAGS.optimizer)

			config = tf.ConfigProto(log_device_placement=self.FLAGS.log_device_placement)
			config.gpu_options.allow_growth = True
			config.gpu_options.per_process_gpu_memory_fraction = self.use_gpu
			if self.model.graph is not None:
				print('worker {} loads pretrained mbnt'.format(self.worker_id))
				sess = tf.Session(graph=self.model.graph, config=config)
				
				self.model.variables.set_session(sess)

				if self.FLAGS.optimizer == 'fedprox':
					with self.model.graph.as_default():
						init = tf.global_variables_initializer()
						sess.run(init)
						
				self.model.init_fn(sess)

				if self.FLAGS.optimizer != 'fedprox':
					sess.run(self.model.training_variables_init)
			else:
				sess = tf.Session(config=config)
				init = tf.global_variables_initializer()
			
				self.model.variables.set_session(sess)
				sess.run(init)

	def init_clients(self):
		self.client = Client(self.FLAGS, self.model, modelname=self.FLAGS.modelname)
		
	def load_data(self):
		try:
			print('loading client data with pickle...')
			t = time.time()
			self.client_data_raw = dict(
				zip(self.client_ids, 
					[
						{
						'measures': load('%s/measures_%s'%(self.FLAGS.client_data_path,client_id)), 
						'labels': load('%s/labels_%s'%(self.FLAGS.client_data_path,client_id))
						} 
					for client_id in self.client_ids]))
			print('loading data time:', time.time()-t)

			if self.FLAGS.shuffle_once:
				np.random.seed(10)

				for client_id in self.client_ids:
					perm = np.arange(self.client_data_raw[client_id]['measures'].shape[0])
					np.random.shuffle(perm)
					self.client_data_raw[client_id]['measures'] = self.client_data_raw[client_id]['measures'][perm,]
					self.client_data_raw[client_id]['labels'] = self.client_data_raw[client_id]['labels'][perm,]

		except:
			print('loading client data...')
			t = time.time()
			self.client_data_raw = dict(
				zip(self.client_ids, 
					[
						{
						'measures': np.loadtxt('%s/measures_%s'%(self.FLAGS.client_data_path,client_id), dtype=np.float32,delimiter=','),
						'labels': np.loadtxt('%s/labels_%s'%(self.FLAGS.client_data_path,client_id), dtype=np.float32,delimiter=',')
						} 
					for client_id in self.client_ids]))
			print('loading data time:', time.time()-t)

		self.load_client_data()

	def load_client_data(self):			
		if self.FLAGS.modelname == 'mbnt':
			self.client_data_tr_tsfm = dict(
				zip(self.tr_client_ids, 
					[cifar10_dataloader(
						self.client_data_raw[client_id]['measures'],
						self.client_data_raw[client_id]['labels'],
						self.tr_transforming, self.batch_size,
						n_workers=self.FLAGS.n_data_workers)
				for client_id in self.tr_client_ids]))

			self.client_data_val_tsfm = dict(
				zip(self.val_te_client_ids, 
					[cifar10_dataloader(
						self.client_data_raw[client_id]['measures'],
						self.client_data_raw[client_id]['labels'],
						self.val_transforming, self.batch_size, 
						n_workers=self.FLAGS.n_data_workers)
				for client_id in self.val_te_client_ids]))
		else:
			one_hot = True
			if self.FLAGS.modelname == 'lstm':
				one_hot = False

			features = None
			if self.FLAGS.numerical_only:
				features = list(range(12))
			
			self.client_data = dict(
				zip(self.client_ids, 
					[Data_loader('','',
						measures=self.client_data_raw[client_id]['measures'], 
						labels=self.client_data_raw[client_id]['labels'], 
						shuffle=self.FLAGS.shuffle,
						imputation=self.FLAGS.imputation,
						one_hot=one_hot,
						features=features) 
					for client_id in self.client_ids]))

		normalizing_feature_selector = None
		if self.FLAGS.num_normalized_features is not None:
			normalizing_feature_selector = np.repeat(False, self.FLAGS.feature_size)
			normalizing_feature_selector[:self.FLAGS.num_normalized_features] = True

		if self.FLAGS.global_normalizing:
			train_measures = load(self.FLAGS.client_data_path+'/fold%s_measures_tr.pkl'%self.FLAGS.fold)
			col_mean = train_measures.mean(axis=0)
			col_std = train_measures.std(axis=0)
		else:
			col_mean = None
			col_std = None

		if self.FLAGS.normalizing and self.FLAGS.modelname != 'mbnt':
			for ci, data in self.client_data.items():
				data.normalize(col_mean=col_mean, col_std=col_std, cols=normalizing_feature_selector)

	def set_client_ids(self, client_ids):
		self.cur_client_ids = client_ids

	def set_global_weights(self, weights):
		for client_id in self.clients:
			self.clients[client_id].set_global_weights(weights)

	def get_weights(self):
		if self.FLAGS.modelname == 'mbnt':
			return self.model.variables.get_weights()

		return self.model.variables.get_flat()

	def train_centralized(self, weights):
		print('Worker {}\'s training client {}'.format(
			self.worker_id,
			','.join([str(cid) for cid in self.cur_client_ids])))

		if self.FLAGS.modelname == 'mbnt':
			combined_measures = np.concatenate([self.client_data_raw[client_id]['measures'] for client_id in self.cur_client_ids])
			combined_labels = np.concatenate([self.client_data_raw[client_id]['labels'] for client_id in self.cur_client_ids])

			client_data = cifar10_dataloader(
				combined_measures,
				combined_labels,
				self.tr_transforming, self.batch_size,
				n_workers=self.FLAGS.n_data_workers, shuffle=True)
			size = combined_measures.shape[0]
		else:
			one_hot = True
			if self.FLAGS.modelname == 'lstm':
				one_hot = False

			client_data = Data_loader('','',
						measures=self.client_data[self.cur_client_ids[0]].measures, 
						labels=self.client_data[self.cur_client_ids[0]].labels, 
						shuffle=self.FLAGS.shuffle,
						imputation=self.FLAGS.imputation,
						one_hot=one_hot)

			for i in range(1, len(self.cur_client_ids)):
				client_data.append(self.client_data[self.cur_client_ids[i]])

			client_data.shuffle_once()

			size = client_data.num_examples
			print('centralized size:', size)
			
		self.client.init_client(self.cur_client_ids[0], client_data, size, 
			True, self.learning_rate)

		updated_weights = self.client.update_weights(weights)

		return updated_weights

	def train_on_clients(self, weights, return_sizes=True, return_grad_norm=None, grad_only=False):
		updated_weights_lst = []
		trained_client_ids = []
		client_sample_sizes = []
		grad_norm_lst = []
		
		print('Worker {}\'s training client {}'.format(
			self.worker_id,
			','.join([str(cid) for cid in self.cur_client_ids])))

		self.size_sum = 0
		for ci in range(len(self.cur_client_ids)):
			client_id = self.cur_client_ids[ci]

			if isinstance(weights,list):
				w = weights[ci]
			else:
				w = weights

			if self.FLAGS.modelname == 'mbnt':
				client_data = self.client_data_tr_tsfm[client_id]
			else:
				client_data = self.client_data[client_id]

			self.client.init_client(client_id, client_data, 
				self.client_data_raw[client_id]['measures'].shape[0], True,
				self.learning_rate)
			self.size_sum += self.client_data_raw[client_id]['measures'].shape[0]
			
			if return_grad_norm is not None:
				updated_weights, grad_norm = self.client.update_weights(w, return_grad_norm=return_grad_norm, grad_only=grad_only)#(weights_lst[i])
				grad_norm_lst.append(grad_norm)
			else:
				updated_weights = self.client.update_weights(w)#(weights_lst[i])

			updated_weights_lst.append(updated_weights)
			trained_client_ids.append(client_id)
			client_sample_sizes.append(self.client.size)

		print('fed size:', self.size_sum)

		if return_grad_norm is not None:
			return list(zip(updated_weights_lst, grad_norm_lst, trained_client_ids))

		if return_sizes:
			return list(zip(updated_weights_lst,trained_client_ids,client_sample_sizes))
		return updated_weights_lst

	def compute_scores(self, weights, score='F1', split='train', average='weighted'):
		print('compute_scores clients:', self.cur_client_ids)
		
		scores = []
		if self.FLAGS.modelname != 'mbnt':
			clients_data = self.client_data
			is_training = None
		else:
			if split == 'train':
				is_training = True
				clients_data = self.client_data_tr_tsfm
			else:
				is_training = False
				clients_data = self.client_data_val_tsfm

		for client_id in self.cur_client_ids:
			self.client.init_client(client_id, clients_data[client_id], 
				self.client_data_raw[client_id]['measures'].shape[0],
				is_training, self.learning_rate)

			if score == 'F1':
				scores.append(self.client.compute_f1(weights, average=average))
			elif score == 'acc':
				scores.append(self.client.compute_acc(weights))
			elif score == 'auc':
				scores.append(self.client.compute_auc(weights))
			elif score == 'loss':
				scores.append(self.client.compute_loss(weights))
			else:
				scores.append(0)

		return scores

	def compute_scores_losses(self, weights, score='F1', average='weighted', loss=True, split='train'):
		print('compute_scores_losses clients:', self.cur_client_ids)

		scores = []
		losses = []

		if self.FLAGS.modelname != 'mbnt':
			clients_data = self.client_data
			is_training = None
		else:
			if split == 'train':
				is_training = True
				clients_data = self.client_data_tr_tsfm
			else:
				is_training = False
				clients_data = self.client_data_val_tsfm

		for client_id in self.cur_client_ids:
			
			self.client.init_client(client_id, clients_data[client_id], 
				self.client_data_raw[client_id]['measures'].shape[0],
				is_training, self.learning_rate)

			if score == 'F1':
				scores.append(self.client.compute_f1(weights, average=average))
			elif score == 'acc':
				scores.append(self.client.compute_acc(weights))
			elif score == 'auc':
				scores.append(self.client.compute_auc(weights))

			if loss:
				losses.append(self.client.compute_loss(weights))
		
		if loss:
			return list(zip(scores, losses))
		return scores

	def set_learning_rate(self, learning_rate):
		self.learning_rate = learning_rate

	def init_client_graphs(self):
		if self.FLAGS.modelname == 'mbnt':
			self.save_model()
		
		self.init_graph()

		if self.FLAGS.modelname == 'mbnt':
			self.restore_model()

		self.init_clients()

		return True

	def save_model(self, best=False):
		if not best:
			fn = self.save_fn
		else:
			fn = self.save_fn_best
		self.model.saver.save(self.model.variables.sess, fn)

	def restore_model(self, best=False):
		if not os.path.isfile(self.save_fn+'.index') and self.FLAGS.resume:
			if not best:
				fn = self.FLAGS.resume_path+'/worker_{}.ckpt'.format(self.worker_id)
			else:
				fn = self.FLAGS.resume_path+'/worker_{}_best.ckpt'.format(self.worker_id)
		else:
			if not best:
				fn = self.save_fn
			else:
				fn = self.save_fn_best

		if os.path.isfile(fn+'.index'):
			self.model.saver.restore(self.model.variables.sess, fn)

	def _print_class_balance(self):
		with open('{}/data_balance_log.txt'.format(self.FLAGS.out_path),'a') as f:
			for client_id in self.client_ids:
				print('client {}:'.format(client_id), file=f)
				labels = np.argmax(self.client_data[client_id].labels, axis=1)
				for c in range(self.FLAGS.num_classes):
					print('  size of class {}: {}'.format(c, sum(labels==c)), file=f)

