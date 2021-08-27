from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import math

import tensorflow as tf

from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc, roc_auc_score

import time

import sys
from utils import load, dump

import os
import random

class Client(object):
	def __init__(self, FLAGS, model, modelname='FFN'):

		self.FLAGS = FLAGS
		self.model = model
		self.data = {}
		self.modelname = modelname
		self.picked_mini_batches_idx = None
		self.last_global_weights = None

	def init_client(self, client_id, data, size, is_training, learning_rate):
		self.client_id = client_id
		self.data = data
		self.size = size
		self.is_training = is_training
		self.save_fn = self.FLAGS.out_path+'/model.ckpt'

		if self.FLAGS.batch_size == -1:
			self.bs = self.size
		else:
			self.bs = self.FLAGS.batch_size

		if self.FLAGS.round_robin:
			self.iterations = math.ceil(self.size * self.FLAGS.num_local_epochs / self.bs)
			self.val_iterations = math.ceil(self.size * self.FLAGS.num_local_epochs / self.FLAGS.val_batch_size)
		else:
			self.iterations = self.FLAGS.num_local_epochs * math.ceil(self.size / self.bs)
			self.val_iterations = self.FLAGS.num_local_epochs * math.ceil(self.size / self.FLAGS.val_batch_size)

		self.model.set_learning_rate(learning_rate)

	def set_learning_rate(self, learning_rate):
		self.model.set_learning_rate(learning_rate)

	def set_global_weights(self, weights):
		self.last_global_weights = weights

	def _compute_updates_batch(self, xs, ys):
		self.model.compute_update(self.model.variables.sess, xs, ys)

	def update_weights(self, weights, return_grad_norm=None, grad_only=False):
		if self.FLAGS.optimizer == 'adam':
			self._restore_model()
		
		if self.last_global_weights is not None:
			with open('{}/last_global_weights_error_log.txt'.format(self.FLAGS.out_path),'a') as f:
				print('client {} calculates gradient norm on global weights\n'.format(self.client_id), file=f)

			self.model.variables.set_flat(self.last_global_weights)
		else:
			if self.modelname == 'mbnt':
				self.model.variables.set_weights(weights)
			else:
				self.model.variables.set_flat(weights)
				
		if return_grad_norm == 0:
			grad_norm_squred = self.compute_gradients_norm(datapoint='full')
			self.data.rewind()
			
		if return_grad_norm == 6:
			grad_norm_squred = 0

			n_mini_bathes = math.ceil(self.size / self.bs)
			for i in range(n_mini_bathes):
				grad_norm_squred += self.compute_gradients_norm(datapoint='one_mini_batch')
			grad_norm_squred /= n_mini_bathes
			self.data.rewind()

		if return_grad_norm == 3:
			# first n mini-batches
			self.last_xs, self.last_ys = self.data.next_batch(
				int(self.FLAGS.n_mini_batch_grad_norm*self.FLAGS.batch_size), 
				round_robin=self.FLAGS.round_robin)
			grad_norm_squred = self.compute_gradients_norm()
			self.data.rewind()

		if return_grad_norm == 4 or return_grad_norm == 5:
			num_mini_batches = math.ceil(self.size / self.bs)

			if self.picked_mini_batches_idx is None or return_grad_norm == 5:
				self.picked_mini_batches_idx = np.random.choice(range(num_mini_batches), self.FLAGS.n_mini_batch_grad_norm, replace=False)

			xs = []
			ys = []
			for i in range(num_mini_batches):
				self.last_xs, self.last_ys = self.data.next_batch(self.FLAGS.batch_size, round_robin=self.FLAGS.round_robin)
					
				if i in self.picked_mini_batches_idx:
					xs.append(self.last_xs)
					ys.append(self.last_ys)

			self.last_xs = np.concatenate(xs, axis=0)
			self.last_ys = np.concatenate(ys, axis=0)
			grad_norm_squred = self.compute_gradients_norm()
			self.data.rewind()

		if not grad_only:
			if self.modelname == 'mbnt':
				for b in self.data:
					xs, ys = b['imgs'], b['labels']
					self._compute_updates_batch(xs, ys)
			else:
				for i in range(self.iterations):
					xs, ys = self.data.next_batch(self.FLAGS.batch_size, round_robin=self.FLAGS.round_robin)

					self._compute_updates_batch(xs, ys)
					
		if self.FLAGS.optimizer == 'adam':
			self._save_model()

		if return_grad_norm == 1:
			grad_norm_squred = self.compute_gradients_norm(datapoint='full')
		if return_grad_norm == 2 or return_grad_norm == 7:
			grad_norm_squred = 0

			if self.modelname == 'mbnt':
				n_mini_bathes = 0
				for b in self.data:
					self.last_xs, self.last_ys = b['imgs'], b['labels']
					grad_norm_squred += self.compute_gradients_norm(datapoint='last')
					n_mini_bathes += 1
			else:
				n_mini_bathes = math.ceil(self.size / self.bs)
				for i in range(n_mini_bathes):
					grad_norm_squred += self.compute_gradients_norm(datapoint='one_mini_batch')

			if return_grad_norm == 2:
				grad_norm_squred /= n_mini_bathes
				
		if grad_only:
			return None, grad_norm_squred

		if self.modelname != 'mbnt':
			weights = self.model.variables.get_flat()
		else:
			weights = self.model.variables.get_weights()
			
		if return_grad_norm is not None:
			return weights, grad_norm_squred
		
		return weights

	def compute_f1(self, weights, average='weighted'):
		if self.modelname == 'mbnt':
			self.model.variables.set_weights(weights)

			y_pred = []
			y_true = []
			for b in self.data:
				xs, ys = b['imgs'], b['labels']
				y_pred.extend(np.argmax(self.model.predict(self.model.variables.sess, xs, self.is_training), 1))
				y_true.extend(np.argmax(ys,1))
		else:
			self.model.variables.set_flat(weights)

			xs, ys = self.data.measures, self.data.labels
			y_p = self.model.predict(self.model.variables.sess, xs)
			
			y_pred = np.argmax(y_p, 1)
			y_true = np.argmax(ys, 1)
		
		if self.FLAGS.num_classes > 2 and average == 'binary':
			return 0

		self.curr_test_score = f1_score(y_true, y_pred, average=average)
		return self.curr_test_score

	def compute_acc(self, weights):
		if self.modelname == 'mbnt':
			self.model.variables.set_weights(weights)
			
			y_pred = []
			y_true = []
			for b in self.data:
				xs, ys = b['imgs'], b['labels']
				y_pred.extend(np.argmax(self.model.predict(self.model.variables.sess, xs, self.is_training), 1))
				y_true.extend(np.argmax(ys,1))
		elif self.modelname == 'lstm':
			self.model.variables.set_flat(weights)

			y_pred = []
			y_true = []

			for i in range(self.val_iterations):
				
				xs, ys = self.data.next_batch(self.FLAGS.val_batch_size, round_robin=self.FLAGS.round_robin)
				y_p, ys = self.model.predict(self.model.variables.sess, xs, ys)

				y_pred.extend(np.argmax(y_p, 1))
				y_true.extend(np.argmax(ys,1))
		else:
			self.model.variables.set_flat(weights)

			xs, ys = self.data.measures, self.data.labels
			y_p = self.model.predict(self.model.variables.sess, xs)
			
			y_pred = np.argmax(y_p, 1)
			y_true = np.argmax(ys, 1)
		
		self.curr_test_score = accuracy_score(y_true, y_pred)
		return self.curr_test_score

	def compute_auc(self, weights):
		if self.modelname == 'mbnt':
			self.model.variables.set_weights(weights)
		else:
			self.model.variables.set_flat(weights)
		
		xs, ys = self.data.measures, self.data.labels
		y_p = self.model.predict(self.model.variables.sess, xs)
		
		y_pred = y_p[:,1]
		y_true = np.argmax(ys, 1)

		try:
			if self.FLAGS.num_classes > 2:
				self.curr_test_score = roc_auc_score(y_true, y_p, multi_class='ovr')
			else:
				self.curr_test_score = roc_auc_score(y_true, y_pred)
		except:
			return 0
		return self.curr_test_score

	def compute_loss(self, weights):
		if self.modelname == 'mbnt':
			self.model.variables.set_weights(weights)

			total_loss = 0
			bi = 0
			for b in self.data:
				xs, ys = b['imgs'], b['labels']

				loss = self.model.compute_loss(self.model.variables.sess, xs, ys, is_training=self.is_training)
				total_loss += loss
			return total_loss/len(self.data)

		if self.modelname == 'lstm':
			self.model.variables.set_flat(weights)
			total_loss = 0
			for i in range(self.val_iterations):
				xs, ys = self.data.next_batch(self.FLAGS.val_batch_size, round_robin=self.FLAGS.round_robin)
				loss = self.model.compute_loss(self.model.variables.sess, xs, ys)
				total_loss += (loss * xs.shape[0])
			return total_loss/self.data.num_examples

		self.model.variables.set_flat(weights)
		xs, ys = self.data.measures, self.data.labels
		return self.model.compute_loss(self.model.variables.sess, xs, ys)

	def norm_squared(self, v):
		try:
			gnorm = np.linalg.norm(v.values.flatten())**2
		except:
			gnorm = np.linalg.norm(v.flatten())**2
		return gnorm

	def compute_gradients_norm(self, weights=None, datapoint='last'):
		if weights is not None:
			if self.modelname == 'mbnt':
				self.model.variables.set_weights(weights)
			else:
				self.model.variables.set_flat(weights)

		if datapoint=='full':
			xs, ys = self.data.measures, self.data.labels
			if self.modelname == 'lstm':
				grad_norm_squred = 0
				
				n_mini_bathes = math.ceil(self.size / self.bs)
				for i in range(n_mini_bathes):
					xs, ys = self.data.next_batch(self.bs, round_robin=self.FLAGS.round_robin)
					gradients = self.model.compute_gradients(self.model.variables.sess, xs, ys)
					grad_norm_squred += np.sum([self.norm_squared(grads) for grads in gradients])
				return grad_norm_squred / n_mini_bathes

		elif datapoint=='one':
			xs, ys = self.data.next_batch(1, round_robin=self.FLAGS.round_robin)
		elif datapoint=='one_mini_batch':
			xs, ys = self.data.next_batch(self.bs, round_robin=self.FLAGS.round_robin)
		else:
			xs, ys = self.last_xs, self.last_ys
			
		gradients = self.model.compute_gradients(self.model.variables.sess, xs, ys)
		grad_norm_squred = np.sum([self.norm_squared(grads) for grads in gradients])
		
		return grad_norm_squred
		