from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

import ray
import ray.experimental.tf_utils

import os
import sys
import random

from language_utils import letter_to_vec, word_to_indices

try:
	import mobilenet_v2
except:
	print('not importing mobilenet_v2')

pretrained_mobilenet_path='mobilenet_checkpoints/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.ckpt'

class LogisticRegression(object):
	def __init__(self, learning_rate, feature_size, num_classes):
		self.learning_rate = learning_rate
		self.feature_size = feature_size
		self.num_classes = num_classes
		self.graph = None

	def set_learning_rate(self, learning_rate):
		self.learning_rate = learning_rate

	def build_graph(self, optimizer='adam'):
		graph_level_seed = 1
		tf.reset_default_graph()
		tf.set_random_seed(graph_level_seed)

		self.x = tf.placeholder(tf.float32, [None, self.feature_size])
		self.y = tf.placeholder(tf.float32, [None, self.num_classes])
		self.learning_rate_tensor = tf.placeholder(tf.float32, shape=[], name="learning_rate")

		self.W = tf.Variable(tf.zeros([self.feature_size, self.num_classes]))
		self.b = tf.Variable(tf.zeros([self.num_classes]))

		self.y_linear = tf.matmul(self.x, self.W) + self.b
		self.pred = tf.nn.softmax(self.y_linear)

		with tf.name_scope('loss'):
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
			labels=self.y, logits=self.y_linear)

		self.cross_entropy = tf.reduce_mean(cross_entropy) + 0.01 * tf.nn.l2_loss(self.W)

		if optimizer == 'sgd':
			self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate_tensor)
		elif optimizer == 'adam':
			self.optimizer = tf.train.AdamOptimizer(self.learning_rate_tensor)
		
		self.train_step = self.optimizer.minimize(self.cross_entropy)

		with tf.name_scope('accuracy'):
			correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		
		self.variables = ray.experimental.tf_utils.TensorFlowVariables(
			self.cross_entropy)#, self.sess)

		self.grads_and_vars = self.optimizer.compute_gradients(self.cross_entropy)

		self.grads_and_vars_placeholder = [(tf.placeholder(
			tf.float32, shape=gv[1].get_shape()), gv[1])
								for gv in self.grads_and_vars]

		self.apply_grads_placeholder = self.optimizer.apply_gradients(
			self.grads_and_vars_placeholder)
		self.saver = tf.train.Saver()

	def compute_gradients(self, sess, x, y):
		return sess.run(
			[gv[0] for gv in self.grads_and_vars],
			feed_dict={
				self.x: x,
				self.y: y,
				self.learning_rate_tensor: self.learning_rate
			})

	def apply_gradients(self, sess, gradients):
		feed_dict = {self.learning_rate_tensor: self.learning_rate}
		for i in range(len(self.grads_and_vars_placeholder)):
			feed_dict[self.grads_and_vars_placeholder[i][0]] = gradients[i]
		sess.run(self.apply_grads_placeholder, feed_dict=feed_dict)

	def predict(self, sess, x):
		return sess.run(self.pred, feed_dict={self.x: x})

	def compute_accuracy(self, sess, x, y):
		return sess.run(self.accuracy,
			feed_dict={
				self.x: x,
				self.y: y
			})

	def compute_loss(self, sess, x, y):
		return sess.run(self.cross_entropy,
			feed_dict={
				self.x: x,
				self.y: y
			})

	def compute_update(self, sess, x, y):
		sess.run(self.train_step, feed_dict={self.x: x, self.y: y, self.learning_rate_tensor: self.learning_rate})

class FFN(object):
	def __init__(self, learning_rate, feature_size, num_classes, hidden_size):
		self.learning_rate = learning_rate
		self.feature_size = feature_size
		self.num_classes = num_classes
		self.hidden_size = hidden_size
		self.operation_level_seed = 2
		self.graph = None

	def set_learning_rate(self, learning_rate):
		self.learning_rate = learning_rate
		
	def _linear(self, input_layer, output_size, scope=None, stddev=1.0, bias_init=0.0):
		with tf.variable_scope(scope or 'linear'):
			w = tf.get_variable(
				'w',
				shape=[input_layer.get_shape()[1], output_size],
				dtype=tf.float32,
				initializer=tf.truncated_normal_initializer(stddev=stddev, seed=self.operation_level_seed)
			)
			b = tf.get_variable(
				'b',
				shape=[output_size],
				dtype=tf.float32,
				initializer=tf.constant_initializer(bias_init)
			)
			return tf.matmul(input_layer, w) + b

	def build_graph(self, optimizer='adam'):
		tf.reset_default_graph()
		tf.set_random_seed(1)

		self.x = tf.placeholder(tf.float32, [None, self.feature_size])
		self.y = tf.placeholder(tf.float32, [None, self.num_classes])
		self.learning_rate_tensor = tf.placeholder(tf.float32, shape=[], name="learning_rate")

		h0 = tf.nn.relu(self._linear(self.x, self.hidden_size, scope='h0', stddev=0.02))
		h1 = tf.nn.relu(self._linear(h0, self.hidden_size, scope='h1', stddev=0.02))

		self.final_output = self._linear(h1, self.num_classes, scope='h2', stddev=0.02)

		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			labels=self.y, logits=self.final_output))

		if optimizer == 'sgd':
			self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate_tensor)
		elif optimizer == 'adam':
			self.optimizer = tf.train.AdamOptimizer(self.learning_rate_tensor)
		
		self.train_step = self.optimizer.minimize(self.loss)

		self.variables = ray.experimental.tf_utils.TensorFlowVariables(
			self.loss)

		self.grads_and_vars = self.optimizer.compute_gradients(self.loss)

		self.grads_and_vars_placeholder = [(tf.placeholder(
			tf.float32, shape=gv[1].get_shape()), gv[1])
								for gv in self.grads_and_vars]

		self.apply_grads_placeholder = self.optimizer.apply_gradients(
			self.grads_and_vars_placeholder)

		self.saver = tf.train.Saver()

	def compute_gradients(self, sess, x, y):
		return sess.run(
			[gv[0] for gv in self.grads_and_vars],
			feed_dict={
				self.x: x,
				self.y: y,
				self.learning_rate_tensor: self.learning_rate
			})

	def apply_gradients(self, sess, gradients):
		feed_dict = {self.learning_rate_tensor: self.learning_rate}
		for i in range(len(self.grads_and_vars_placeholder)):
			feed_dict[self.grads_and_vars_placeholder[i][0]] = gradients[i]
		sess.run(self.apply_grads_placeholder, feed_dict=feed_dict)

	def apply_flat_gradients(self, sess, flat_gradients, layer_shapes):
		feed_dict = {self.learning_rate_tensor: self.learning_rate}
		start = 0
		for i in range(len(self.grads_and_vars_placeholder)):
			shape = layer_shapes[i]

			if len(shape) == 1:
				size = shape[0]
			else:
				size = shape[0] * shape[1]

			end = start + size
			gradients = np.reshape(flat_gradients[start:end], shape)

			feed_dict[self.grads_and_vars_placeholder[i][0]] = gradients
			start = end

		sess.run(self.apply_grads_placeholder, feed_dict=feed_dict)

	def predict(self, sess, x):
		return sess.run(tf.nn.softmax(self.final_output), feed_dict={self.x: x})
	
	def compute_update(self, sess, x, y):
		sess.run(self.train_step, feed_dict={self.x: x, self.y: y, self.learning_rate_tensor: self.learning_rate})

	def compute_loss(self, sess, x, y):
		return sess.run(self.loss, feed_dict={self.x: x,self.y: y})

class MobilenetV2(object):
	def __init__(self, learning_rate, num_classes, input_size=224, dropout_keep_prob=0.8):
		self.learning_rate = learning_rate
		self.num_classes = num_classes
		self.dropout_keep_prob = dropout_keep_prob
		self.input_size = input_size
		self.operation_level_seed = 2
		self.graph = None
		self.pretrained_model_path = pretrained_mobilenet_path

	def set_learning_rate(self, learning_rate):
		self.learning_rate = learning_rate
		
	def build_graph(self, optimizer='adam'):
		tf.reset_default_graph()
		self.graph = tf.Graph()

		with self.graph.as_default():
			tf.set_random_seed(1)
			
			self.learning_rate_tensor = tf.placeholder(tf.float32, shape=[], name="learning_rate")
			self.input_tensor = tf.placeholder(tf.float32, shape=[None, self.input_size, self.input_size, 3], name="input_tensor")
			self.label_tensor = tf.placeholder(tf.int32, None)
			self.is_training = tf.placeholder_with_default(False,None,name='is_training')
			self.keep_prob = tf.placeholder(tf.float32,shape=(),name='keep_prob')

			with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(
				is_training=self.is_training,dropout_keep_prob=self.keep_prob)):
					self.logits, self.endpoints = mobilenet_v2.mobilenet(
						self.input_tensor,num_classes=self.num_classes)
			
			variables_to_restore = tf.contrib.framework.get_variables_to_restore(
				exclude=['MobilenetV2/Logits', 'MobilenetV2/Conv_1', 'MobilenetV2/expanded_conv_16'])

			# init function of loading pretrained weights
			self.init_fn = tf.contrib.framework.assign_from_checkpoint_fn(
				self.pretrained_model_path, variables_to_restore)

			# logit variables and its init function
			self.training_variables = []
			self.training_variables.extend(tf.contrib.framework.get_variables('MobilenetV2/Logits'))
			self.training_variables.extend(tf.contrib.framework.get_variables('MobilenetV2/Conv_1'))
			self.training_variables.extend(tf.contrib.framework.get_variables('MobilenetV2/expanded_conv_16'))
			self.training_variables_init = tf.variables_initializer(self.training_variables)

			training_variable_names = [v.name[:-2] for v in self.training_variables]

			tf.losses.sparse_softmax_cross_entropy(labels=self.label_tensor, logits=self.logits)
			self.loss = tf.losses.get_total_loss()

			self.global_init = None
			if optimizer == 'sgd':
				self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate_tensor)
			elif optimizer == 'adam':
				self.optimizer = tf.train.AdamOptimizer(self.learning_rate_tensor)
			
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				self.train_step = self.optimizer.minimize(self.loss, var_list=self.training_variables)#,global_step=self.global_step)
				self.grads_and_vars = self.optimizer.compute_gradients(self.loss, var_list=[tf.contrib.framework.get_variables('MobilenetV2/Logits')])
			
			self.prediction = tf.to_int32(tf.argmax(self.logits, 1))
			self.correct_prediction = tf.equal(self.prediction, self.label_tensor)
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

			self.variables = ray.experimental.tf_utils.TensorFlowVariables(
				self.loss)

			removing_variables = [k for k in self.variables.variables if k not in training_variable_names]
			for k in removing_variables:
				del self.variables.variables[k]

			self.saver = tf.train.Saver()
			
	def compute_gradients(self, sess, x, y):
		if len(y.shape) > 1:
			y = np.argmax(y, 1)

		return sess.run(
			[gv[0] for gv in self.grads_and_vars],
			feed_dict={
				self.input_tensor: x,
				self.label_tensor: y,
				self.is_training: True,
				self.keep_prob:self.dropout_keep_prob,
				self.learning_rate_tensor: self.learning_rate
			})

	def predict(self, sess, x, is_training):
		keep_prob = 1.0 if not is_training else self.dropout_keep_prob
		return sess.run(tf.nn.softmax(self.logits), 
			feed_dict = {self.input_tensor:x,
						self.is_training:is_training,
						self.keep_prob:keep_prob})
	
	def compute_update(self, sess, x, y):
		if len(y.shape) > 1:
			y = np.argmax(y, 1)

		sess.run(self.train_step, 
			feed_dict={self.input_tensor:x,
						self.label_tensor:y,
						self.is_training:True,
						self.keep_prob:self.dropout_keep_prob,
						self.learning_rate_tensor: self.learning_rate})

	def compute_loss(self, sess, x, y, is_training=True):
		keep_prob = 1.0 if not is_training else self.dropout_keep_prob

		if len(y.shape) > 1:
			y = np.argmax(y, 1)
			
		return sess.run(self.loss, 
			feed_dict={self.input_tensor: x,
						self.label_tensor: y,
						self.is_training:is_training,
						self.keep_prob:keep_prob})

class StackedLSTM(object):
	def __init__(self, seq_len, num_classes, n_hidden):
		self.seq_len = seq_len
		self.num_classes = num_classes
		self.n_hidden = n_hidden
		self.graph = None
		
	def set_learning_rate(self, learning_rate):
		self.learning_rate = learning_rate
		
	def build_graph(self, optimizer='sgd', seed=1):
		tf.reset_default_graph()
		tf.set_random_seed(seed)

		self.learning_rate_tensor = tf.placeholder(tf.float32, shape=[], name="learning_rate")
		
		self.input_tensor = tf.placeholder(tf.int32, [None, self.seq_len])
		embedding = tf.get_variable("embedding", [self.num_classes, 8])
		x = tf.nn.embedding_lookup(embedding, self.input_tensor)
		self.label_tensor = tf.placeholder(tf.int32, [None, self.num_classes])

		stacked_lstm = rnn.MultiRNNCell(
			[rnn.BasicLSTMCell(self.n_hidden) for _ in range(2)])
		outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32)
		self.logits = tf.layers.dense(inputs=outputs[:,-1,:], units=self.num_classes)

		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.label_tensor))
		
		if optimizer == 'sgd':
			self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate_tensor)
		elif optimizer == 'adam':
			self.optimizer = tf.train.AdamOptimizer(self.learning_rate_tensor)
		
		self.train_step = self.optimizer.minimize(self.loss)
		
		self.variables = ray.experimental.tf_utils.TensorFlowVariables(
			self.loss)

		self.grads_and_vars = self.optimizer.compute_gradients(self.loss)

		self.saver = tf.train.Saver()

	def compute_gradients(self, sess, x, y):
		x = self.process_x(x)
		y = self.process_y(y)

		return sess.run(
			[gv[0] for gv in self.grads_and_vars],
			feed_dict={
				self.input_tensor: x,
				self.label_tensor: y,
				self.learning_rate_tensor: self.learning_rate
			})
		
	def process_x(self, raw_x_batch):
		x_batch = [word_to_indices(word) for word in raw_x_batch]
		x_batch = np.array(x_batch)
		return x_batch

	def process_y(self, raw_y_batch):
		y_batch = [letter_to_vec(c) for c in raw_y_batch]
		return y_batch

	def predict(self, sess, x, y):
		x = self.process_x(x)
		y = self.process_y(y)

		return sess.run(tf.nn.softmax(self.logits), 
			feed_dict = {self.input_tensor:x}), y
	
	def compute_update(self, sess, x, y):
		x = self.process_x(x)
		y = self.process_y(y)

		sess.run(self.train_step, 
			feed_dict={self.input_tensor:x,
						self.label_tensor:y,
						self.learning_rate_tensor: self.learning_rate})

	def compute_loss(self, sess, x, y):
		x = self.process_x(x)
		y = self.process_y(y)
			
		return sess.run(self.loss, 
			feed_dict={self.input_tensor: x,
						self.label_tensor: y})


