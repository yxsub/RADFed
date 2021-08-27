from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import math
from scipy.special import comb
import itertools
import shutil

import ray
import model
from modelActor import ModelActor

import tensorflow as tf

import time
import re
import pickle

from utils import *
from server_func import *
from testing import *
from arg_parser import arg_parser

import psutil
import gc

import os
import sys

FLAGS = arg_parser().parse_args()

print(FLAGS)

use_gpu = 1/np.ceil((FLAGS.num_tr_workers+1)/FLAGS.num_gpus) if int(FLAGS.num_gpus) > 0 else 0

print('use_gpu:',use_gpu)

if use_gpu:
	os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
				[str(i) for i in FLAGS.gpu_ids])

class ParameterServerWeight(object):
	def __init__(self, modelname='FFN'):
		self.modelname = modelname
		self.coef = 0
		np.random.seed(FLAGS.seed)

	def set_coef(self, coef):
		self.coef = coef

	def set_weights(self, weights):
		self.weights = weights

		if self.modelname == 'mbnt':
			self.weights = self.flatten(weights)

			self.variables = []
			for k in weights:
				val_shape = weights[k].shape
				flatten_val = np.reshape(weights[k], (-1,))
				val_len = len(flatten_val)

				self.variables.append((k, val_shape, val_len))
		
	def get_weights(self):
		if self.modelname != 'mbnt':
			return self.weights

		weights_dict = {}
		i = 0
		for v in self.variables:
			val_name = v[0]
			val_shape = v[1]
			val_len = v[2]

			weights_dict[val_name] = np.reshape(self.weights[i:i+val_len], val_shape)
			i += val_len

		return weights_dict

	def flatten(self, weights):
		return np.concatenate([np.reshape(weights[k], (-1,)) for k in weights])


	def divergence(self, all_weights):
		d = 0
		for w1, w2 in itertools.combinations(all_weights, 2):
			d += (1-np.dot(w1,w2)/(np.linalg.norm(w1)*np.linalg.norm(w2)))
		
		return self.coef * d

	def apply_weights(self, *weights):
		if self.modelname == 'mbnt':
			weights = [self.flatten(w) for w in weights]

		div = 0
		if FLAGS.save_divergence:
			div = self.divergence(weights)

		self.weights = np.mean(weights, axis=0)
		return self.get_weights(), div

	def shuffle_weights(self, *weights):
		weights = list(weights)
		np.random.shuffle(weights)
		return weights

@ray.remote(num_gpus=use_gpu)
class ModelTrainActor(ModelActor):
	def __init__(self, args, worker_id, tr_client_ids, val_client_ids, te_client_ids, num_workers):
		super().__init__(args, worker_id, tr_client_ids, val_client_ids, te_client_ids, num_workers, use_gpu)
		
		if use_gpu:
			os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
				[str(i) for i in ray.get_gpu_ids()])
			print("Training Worker {} is allowed to use GPUs {}.".format(worker_id,ray.get_gpu_ids()))
		
		if args.C == 1:
			num_client_per_worker = len(client_ids) / num_workers
			start = int(num_client_per_worker * worker_id)
			end = int(num_client_per_worker * (worker_id + 1))

			self.client_ids = client_ids[start:end]

		self.init_graph()
		self.load_data()
		self.restore_model()
		self.init_clients()

def send_client_ids_to_workers(workers, client_ids, num_client_per_worker):
	for worker_id in range(len(workers)):
		start = int(num_client_per_worker * worker_id)
		end = int(num_client_per_worker * (worker_id + 1))
		client_ids_on_worker = client_ids[start:end]
		workers[worker_id].set_client_ids.remote(client_ids_on_worker)

def test(workers, client_ids, weights, split='test'):
	num_client_per_worker = len(client_ids) / FLAGS.num_tr_workers
	send_client_ids_to_workers(workers, client_ids, num_client_per_worker)
	
	test_worker_scores_losses = [test_worker.compute_scores_losses.remote(weights, score=FLAGS.score, split=split)
				for test_worker in workers]

	testwscoreslosses = [client_test_scores_losses 
					for testwscoreloss in test_worker_scores_losses 
					for client_test_scores_losses in ray.get(testwscoreloss)]

	test_scores = [client_test_scores_losses[0] for client_test_scores_losses in testwscoreslosses]
	test_losses = [client_test_scores_losses[1] for client_test_scores_losses in testwscoreslosses]

	return test_scores, test_losses

def train():
	if FLAGS.redis_address is None:
		ray.init(num_gpus=FLAGS.num_gpus, redis_max_memory=4*1024**3, object_store_memory=1024**3)
	else:
		ray.init(redis_address=FLAGS.redis_address, redis_max_memory=4*1024**3, object_store_memory=1024**3)

	step_decay = StepDecay(FLAGS.learning_rate, 
		FLAGS.learning_rate_decay_factor, FLAGS.learning_rate_decay_freq)

	ps = ParameterServerWeight(modelname=FLAGS.modelname)

	tr_client_ids = np.loadtxt(os.path.join(FLAGS.client_data_path,'fold{}_tr_client_ids.lst'.format(FLAGS.fold)),dtype=int,delimiter=',')
	val_client_ids = np.loadtxt(os.path.join(FLAGS.client_data_path,'fold{}_val_client_ids.lst'.format(FLAGS.fold)),dtype=int,delimiter=',')
	te_client_ids = np.loadtxt(os.path.join(FLAGS.client_data_path,'fold{}_te_client_ids.lst'.format(FLAGS.fold)),dtype=int,delimiter=',')
	
	if FLAGS.inner_fold is not None:
		tr_client_ids, val_client_ids = get_inner_folds(FLAGS.client_data_path,FLAGS.fold,FLAGS.inner_fold,tr_client_ids,val_client_ids,te_client_ids)

	num_tr_clients = len(tr_client_ids)
	num_val_clients = len(val_client_ids)

	np.random.seed(FLAGS.seed)

	np.random.shuffle(tr_client_ids)
	num_selected_tr_clients = int(max(FLAGS.C * num_tr_clients, 1))
	ps.set_coef(comb(num_selected_tr_clients, 2, exact=False)**-1)

	log_str = ''
	picking_clients_history = ''
	if not FLAGS.resume:
		log_str += 'training clients: {}\n'.format(tr_client_ids)
		log_str += 'validation clients: {}\n'.format(val_client_ids)
		log_str += 'test clients: {}\n'.format(te_client_ids)

	window_size = FLAGS.window_size / (FLAGS.C * FLAGS.num_local_epochs)
	converge_window = FLAGS.converge_window / (FLAGS.C * FLAGS.num_local_epochs)

	# create workers
	workers = [ModelTrainActor.remote(FLAGS, worker_id, tr_client_ids, val_client_ids, te_client_ids, FLAGS.num_tr_workers)
					for worker_id in range(FLAGS.num_tr_workers)]

	if FLAGS.resume:
		print('Loading weights from last trianed model:', FLAGS.resume_path+'/weights.ndarray')
		if FLAGS.modelname == 'mbnt':
			current_weights = load(FLAGS.resume_path+'/weights.dict')
			best_weights = load(FLAGS.resume_path+'/best_weights.dict')
		else:
			current_weights = np.loadtxt(FLAGS.resume_path+'/weights.ndarray', dtype=np.float32, delimiter=',')
			best_weights = np.loadtxt(FLAGS.resume_path+'/best_weights.ndarray', dtype=np.float32, delimiter=',')
		
		ps.set_weights(current_weights)
		
		try:
			client_training_score_history = np.loadtxt(
				FLAGS.resume_path+'/client_training_{}_history.lst'.format(FLAGS.score), dtype=np.float32, delimiter=',').tolist()
			client_validation_score_history = np.loadtxt(
				FLAGS.resume_path+'/client_validation_{}_history.lst'.format(FLAGS.score), dtype=np.float32, delimiter=',').tolist()

			if num_tr_clients == 1:
				client_training_score_history = [client_training_score_history]
			if num_val_clients == 1:
				client_validation_score_history = [client_validation_score_history]

			avg_training_score_history = np.loadtxt(
				FLAGS.resume_path+'/avg_training_{}_history.lst'.format(FLAGS.score), dtype=np.float32, delimiter=',').tolist()
			avg_validation_score_history = np.loadtxt(
				FLAGS.resume_path+'/avg_validation_{}_history.lst'.format(FLAGS.score), dtype=np.float32, delimiter=',').tolist()
			avg_training_loss_history = np.loadtxt(
				FLAGS.resume_path+'/avg_training_loss_history.lst', dtype=np.float32, delimiter=',').tolist()
			avg_validation_loss_history = np.loadtxt(
				FLAGS.resume_path+'/avg_validation_loss_history.lst', dtype=np.float32, delimiter=',').tolist()
		except FileNotFoundError:
			client_validation_score_history = [[] for _ in range(num_val_clients)]
			avg_validation_score_history = []
			avg_validation_loss_history = []

		start_round = FLAGS.start_round
		if start_round == -1:
			start_round = len(avg_training_score_history) // FLAGS.num_shuffle_round

		weights_lst = [current_weights for _ in range(num_selected_tr_clients)]

		if FLAGS.return_grad_norm is not None:
			is_probs = np.loadtxt(FLAGS.resume_path+'/is_probs.lst', dtype=np.float32, delimiter=',').tolist()
			with open(FLAGS.resume_path+'/grad_norm.dict', 'rb') as f:
				grad_norm_dict = pickle.load(f)
			norm_dict_log_str = ''
	
		for i in range(start_round):
			for shuffle_round in range(FLAGS.num_shuffle_round):
				_ = ps.shuffle_weights(*weights_lst)
				
				if FLAGS.return_grad_norm is None:
					_ = np.random.choice(tr_client_ids, num_selected_tr_clients, replace=False)
				else:
					p = is_probs[0]

					_ = np.random.choice(
						tr_client_ids, num_selected_tr_clients, 
						replace=False, p=p)

		best_val_score = max(avg_validation_score_history)
		
		with open(os.path.join(FLAGS.resume_path,'log.txt'),'r') as f:
			lines = f.readlines()
		s = ''.join(lines[-10:])
		converge_cnt = int(re.findall(r'converge_cnt: (\d+)',s)[0])
		curr_max_val_score = float(re.findall(r'curr_max_val.*: (0\.\d+)',s)[0])
		prev_max_val_score = float(re.findall(r'prev_max_val.*: (0\.\d+)',s)[0])
		
	else:
		current_weights = ray.get(workers[0].get_weights.remote())
		ps.set_weights(current_weights)

		client_training_score_history = [[] for _ in range(num_selected_tr_clients)] #range(num_tr_clients)]
		client_validation_score_history = [[] for _ in range(num_val_clients)]
		avg_training_score_history = []
		avg_validation_score_history = []
		avg_training_loss_history = []
		avg_validation_loss_history = []
		divergence_history = []

		is_probs = []

		start_round = 0
		best_val_score = 0

		converge_cnt = 0
		curr_max_val_score = 0
		prev_max_val_score = 0

		if FLAGS.return_grad_norm is not None:
			grad_norm_dict = dict(zip(tr_client_ids, [1.0]*len(tr_client_ids)))
			grad_norm_cnt_dict = dict(zip(tr_client_ids, [0]*len(tr_client_ids)))
			norm_dict_log_str = 'training clients: {}\n'.format(tr_client_ids)

		best_weights = current_weights

	best_round_by_score = 0
	avg_val_loss = 0
	val_score = 0

	def assign_clients_to_worker(worker_id, client_lst, num_client_per_worker):
		start = int(num_client_per_worker * worker_id)
		end = int(num_client_per_worker * (worker_id + 1))
		return client_lst[start:end]

	worker_ids = range(FLAGS.num_tr_workers)

	def plot_and_save():
		np.savetxt(FLAGS.out_path+'/client_training_{}_history.lst'.format(FLAGS.score), client_training_score_history, fmt='%.8f', delimiter=',')
		np.savetxt(FLAGS.out_path+'/avg_training_{}_history.lst'.format(FLAGS.score), avg_training_score_history, fmt='%.8f', delimiter=',')
		np.savetxt(FLAGS.out_path+'/avg_training_loss_history.lst', avg_training_loss_history, fmt='%.8f', delimiter=',')
		np.savetxt(FLAGS.out_path+'/avg_validation_{}_history.lst'.format(FLAGS.score), avg_validation_score_history, fmt='%.8f', delimiter=',')
		np.savetxt(FLAGS.out_path+'/avg_validation_loss_history.lst', avg_validation_loss_history, fmt='%.8f', delimiter=',')
		np.savetxt(FLAGS.out_path+'/client_validation_{}_history.lst'.format(FLAGS.score), client_validation_score_history, fmt='%.8f', delimiter=',')
		
		dump(divergence_history, FLAGS.out_path+'/divergence_history.lst.pkl')
		
		if FLAGS.modelname == 'mbnt':
			dump(current_weights, FLAGS.out_path+'/weights.dict')
		else:
			np.savetxt(FLAGS.out_path+'/weights.ndarray', current_weights, fmt='%.16f', delimiter=',')

	num_client_per_worker = num_tr_clients / FLAGS.num_tr_workers
	weights_lst = [current_weights for _ in range(num_tr_clients)]

	if FLAGS.return_grad_norm is not None:
		for worker_id in worker_ids:
			workers[worker_id].set_client_ids.remote(assign_clients_to_worker(
						worker_id, 
						tr_client_ids, 
						num_client_per_worker))
			if FLAGS.grad_norm_on_global_weights:
				workers[worker_id].set_global_weights.remote(current_weights)

		worker_weights = [workers[worker_id].train_on_clients.remote(
									assign_clients_to_worker(
										worker_id, 
										weights_lst, 
										num_client_per_worker),
									return_sizes=False,
									return_grad_norm=FLAGS.return_grad_norm,
									grad_only=True)
								for worker_id in worker_ids]

		wwgrads = [client_weights_grad_norm 
						for wwgrad in worker_weights 
						for client_weights_grad_norm in ray.get(wwgrad)]
		grad_norm_lst = [client_weights_grad_norm[1] for client_weights_grad_norm in wwgrads]
		client_id_lst = [client_weights_grad_norm[2] for client_weights_grad_norm in wwgrads]

		for cid, grad_norm in zip(client_id_lst, grad_norm_lst):
			grad_norm_dict[cid] = grad_norm
			grad_norm_cnt_dict[cid] += 1

	cur_lr = FLAGS.learning_rate
	for i in range(start_round, FLAGS.num_round):
		
		if FLAGS.return_grad_norm is not None:
			norm_dict_log_str += 'Round {}:\n'.format(i)

		weights_lst = [current_weights for _ in range(num_selected_tr_clients)]
		
		for shuffle_round in range(FLAGS.num_shuffle_round):
			
			cur_step = i * FLAGS.num_shuffle_round + shuffle_round
			if FLAGS.learning_rate_decay_freq is not None:
				cur_lr = step_decay(cur_step)
			
			for worker_id in range(len(workers)):
				workers[worker_id].set_learning_rate.remote(cur_lr)

			current_weights_lst = ps.shuffle_weights(*weights_lst)
			num_client_per_worker = num_selected_tr_clients / FLAGS.num_tr_workers

			if FLAGS.return_grad_norm is None:
				selected_tr_client_ids = np.random.choice(
						tr_client_ids, num_selected_tr_clients, replace=False)
			else:
				grad_norm_sum = sum(grad_norm_dict.values())
				p = [grad_norm_dict[cid]/grad_norm_sum for cid in tr_client_ids]
				
				formated_str = ', '.join(['{:d}:{:0.2f}'.format(k,v) for k,v in grad_norm_dict.items()])
				norm_dict_log_str += 'grad_norm_dict: {{{}}}\n'.format(formated_str)

				formated_str = ', '.join(['{:d}:{:d}'.format(k,v) for k,v in grad_norm_cnt_dict.items()])
				norm_dict_log_str += 'grad_norm_cnt_dict: {{{}}}\n'.format(formated_str)
				
				formated_str = ', '.join(['{:0.4f}'.format(q) for q in p])
				norm_dict_log_str += 'p: [{}]\n'.format(formated_str)

				is_probs.append(p)

				selected_tr_client_ids = np.random.choice(
						tr_client_ids, num_selected_tr_clients, 
						replace=False, p=p)

			picking_clients_history += 'round: {}, shuffle round: {}\n'.format(i, shuffle_round)
			picking_clients_history += 'selected_tr_client_ids: {}\n'.format(','.join([str(cid) for cid in sorted(selected_tr_client_ids)]))
			print('round: {}, shuffle round: {}'.format(i, shuffle_round))
			print('selected_tr_client_ids: {}'.format(','.join([str(cid) for cid in sorted(selected_tr_client_ids)])))

			for worker_id in worker_ids:
				workers[worker_id].set_client_ids.remote(assign_clients_to_worker(
							worker_id, 
							selected_tr_client_ids, 
							num_client_per_worker))

			worker_weights = [workers[worker_id].train_on_clients.remote(
									assign_clients_to_worker(
										worker_id, 
										current_weights_lst, 
										num_client_per_worker),
									return_sizes=False,
									return_grad_norm=FLAGS.return_grad_norm) 
								for worker_id in worker_ids]

			if FLAGS.return_grad_norm is not None:
				wwgrads = [client_weights_grad_norm 
								for wwgrad in worker_weights 
								for client_weights_grad_norm in ray.get(wwgrad)]
				weights_lst = [client_weights_grad_norm[0] for client_weights_grad_norm in wwgrads]
				grad_norm_lst = [client_weights_grad_norm[1] for client_weights_grad_norm in wwgrads]
				client_id_lst = [client_weights_grad_norm[2] for client_weights_grad_norm in wwgrads]

				for cid, grad_norm in zip(client_id_lst, grad_norm_lst):

					if FLAGS.ISalpha is not None:
						grad_norm_dict[cid] = (1 - FLAGS.ISalpha) * grad_norm_dict[cid] + FLAGS.ISalpha * grad_norm
					else:
						sum_grad_norm = grad_norm_dict[cid]*grad_norm_cnt_dict[cid]
						grad_norm_cnt_dict[cid] += 1
						grad_norm_dict[cid] = (sum_grad_norm + grad_norm) / grad_norm_cnt_dict[cid]
			else:
				weights_lst = [client_weights for ww in worker_weights for client_weights in ray.get(ww)]

			if ((i * FLAGS.num_shuffle_round + shuffle_round) +1) % FLAGS.apply_weights_freq == 0:
				current_weights, div = ps.apply_weights(*weights_lst)

				if FLAGS.save_divergence:
					divergence_history.append(div)
					print(ps.coef, div)

			#
			# evaluate on training data
			#
			if ((i * FLAGS.num_shuffle_round + shuffle_round) +1) % FLAGS.eval_freq == 0:
				
				worker_scores = [worker.compute_scores.remote(current_weights, score=FLAGS.score, split='train')
						for worker in workers]
				scores = [client_scores for wscore in worker_scores  for client_scores in ray.get(wscore)]

				num_client_per_worker = len(tr_client_ids) / FLAGS.num_tr_workers
				send_client_ids_to_workers(workers, tr_client_ids, num_client_per_worker)
				worker_losses = [worker.compute_scores.remote(current_weights, score='loss', split='train')
						for worker in workers]
				losses = [client_losses for wloss in worker_losses for client_losses in ray.get(wloss)]

				for j in range(num_selected_tr_clients):
					client_training_score_history[j].append(scores[j])

				avg_tr_score = np.mean(scores)
				avg_tr_loss = np.mean(losses)

				avg_training_score_history.append(avg_tr_score)
				avg_training_loss_history.append(avg_tr_loss)

				log_str += 'Round {}:\n'.format(i*FLAGS.num_shuffle_round+shuffle_round)
				log_str += 'learning_rate: {:.8f}\n'.format(cur_lr)
				log_str += 'Avg training {}: {:.8f}\n'.format(FLAGS.score,avg_tr_score)
				log_str += 'Avg training loss: {:.8f}\n'.format(avg_tr_loss)
				
				val_scores, val_losses = test(workers, val_client_ids, current_weights, split='val')

				avg_val_score = np.mean(val_scores)
				avg_val_loss = np.mean(val_losses)
				
				avg_validation_score_history.append(avg_val_score)
				avg_validation_loss_history.append(avg_val_loss)
				
				for j in range(num_val_clients):
					client_validation_score_history[j].append(val_scores[j])

				curr_round = i * FLAGS.num_shuffle_round + shuffle_round
				curr_max_val_score = max(curr_max_val_score, avg_val_score)

				eval_step = (curr_round+1)//FLAGS.eval_freq

				if eval_step >= window_size:
					prev_max_val_score = max(prev_max_val_score, avg_validation_score_history[int(eval_step-window_size)])
					if (curr_max_val_score - prev_max_val_score) / prev_max_val_score <= FLAGS.threshold:
						converge_cnt += 1
					else:
						converge_cnt = 0

				log_str += 'Avg validation {}: {:.8f}\n'.format(FLAGS.score,avg_val_score)
				log_str += 'Avg validation loss: {:.8f}\n'.format(avg_val_loss)
				log_str += 'curr_max_val_{}: {:.8f}\n'.format(FLAGS.score,curr_max_val_score)
				log_str += 'prev_max_val_{}: {:.8f}\n'.format(FLAGS.score,prev_max_val_score)
				log_str += 'relative {} increase: {:.8f}\n'.format(FLAGS.score,(curr_max_val_score - prev_max_val_score) / prev_max_val_score if prev_max_val_score else 0)
				log_str += 'converge_cnt: {:d}\n'.format(converge_cnt)
				
				val_score = avg_val_score
				if val_score >= best_val_score:
					best_val_score = val_score
					best_weights = current_weights

					if FLAGS.modelname == 'mbnt':
						dump(best_weights, FLAGS.out_path+'/best_weights.dict')
						for w in workers:
							ray.get(w.save_model.remote(best=True))
					
					best_round_by_score = i

			if ((i * FLAGS.num_shuffle_round + shuffle_round) +1) % FLAGS.re_init_client_freq == 0 and i > 0:
				for worker in workers:
					ray.get(worker.init_client_graphs.remote())

				gc.collect()
				
			if FLAGS.save_file and (((i * FLAGS.num_shuffle_round + shuffle_round) +1) % FLAGS.write_freq == 0 or (i+1)==FLAGS.num_round):
				
				if FLAGS.return_grad_norm is not None:
					with open('{}/norm_dict_log.txt'.format(FLAGS.out_path), 'a') as f:
						print(norm_dict_log_str, file=f)
						norm_dict_log_str = ''

					np.savetxt(FLAGS.out_path+'/is_probs.lst', is_probs, fmt='%.8f', delimiter=',')
					with open(FLAGS.out_path+'/grad_norm.dict', 'wb') as f:
						pickle.dump(grad_norm_dict, f)

				with open('{}/log.txt'.format(FLAGS.out_path),'a') as f:
					print(log_str, file=f)
					log_str = ''

				with open('{}/picking_clients_history.txt'.format(FLAGS.out_path),'a') as f:
					print(picking_clients_history, file=f)
					picking_clients_history = ''

				plot_and_save()

		if FLAGS.move_out_path is not None and (i+1) % FLAGS.copy_file_freq == 0:
			try:
				copydir(FLAGS.out_path, FLAGS.move_out_path)
			except Exception as e:
				print(e)

		if FLAGS.stop_round != 0:
			if (i+1) >= FLAGS.stop_round or np.isnan(avg_val_loss):
				plot_and_save()
				break
		else:
			if converge_cnt >= converge_window or np.isnan(avg_val_loss):
				plot_and_save()
				break

	with open('{}/log.txt'.format(FLAGS.out_path),'a') as f:
		print(log_str, file=f)

	if FLAGS.return_grad_norm is not None:
		with open('{}/norm_dict_log.txt'.format(FLAGS.out_path), 'a') as f:
			print(norm_dict_log_str, file=f)

	with open('{}/picking_clients_history.txt'.format(FLAGS.out_path),'a') as f:
		print(picking_clients_history, file=f)

	if FLAGS.modelname != 'mbnt':
		np.savetxt(FLAGS.out_path+'/best_weights.ndarray', best_weights, fmt='%.16f', delimiter=',')

	#
	# test
	#
	if FLAGS.modelname == 'mbnt':
		for w in workers:
			ray.get(w.restore_model.remote(best=True))

	test_scores, test_losses = test(workers, te_client_ids, best_weights, split='test')
	avg_test_score = np.mean(test_scores)
	avg_test_loss = np.mean(test_losses)
	
	if FLAGS.save_file:
		with open('{}/results.txt'.format(FLAGS.out_path),'w') as f:
			print('Best round by {}: {:.8f}'.format(FLAGS.score,best_round_by_score), file=f)
			print('Best Avg validation {}: {:.8f}'.format(FLAGS.score,best_val_score), file=f)
			print('Avg test {}: {:.8f}'.format(FLAGS.score,avg_test_score), file=f)
			print('Avg test loss: {:.8f}'.format(avg_test_loss), file=f)
			print('Best Avg validation {} by last round: {:.8f}'.format(FLAGS.score,val_score), file=f)

	if FLAGS.move_out_path is not None:
		try:
			copydir(FLAGS.out_path, FLAGS.move_out_path)
		except:
			pass

def test_multiple_scores():
	if FLAGS.redis_address is None:
		ray.init(num_gpus=FLAGS.num_gpus)
	else:
		ray.init(redis_address=FLAGS.redis_address)

	te_client_ids = np.loadtxt(os.path.join(FLAGS.client_data_path,'fold{}_te_client_ids.lst'.format(FLAGS.fold)),dtype=int,delimiter=',')
	test_workers = [ModelValidationActor.remote(FLAGS, worker_id, te_client_ids, FLAGS.num_te_workers) 
						for worker_id in range(FLAGS.num_te_workers)]

	test_model(test_workers, FLAGS)

if __name__ == '__main__':
	if FLAGS.test:
		test_multiple_scores()
	else:
		if FLAGS.move_out_path is None or not os.path.isdir(FLAGS.move_out_path):
			if FLAGS.resume:
				os.makedirs(FLAGS.out_path)
				with open('{}/args.txt'.format(FLAGS.out_path),'w') as f:
					print(FLAGS, file=f)
			else:
				while True:
					try:
						os.makedirs(FLAGS.out_path)
						with open('{}/args.txt'.format(FLAGS.out_path),'w') as f:
							print(FLAGS, file=f)
						break
					except FileExistsError:
						if os.path.isfile(FLAGS.out_path+'/weights.ndarray'):
							FLAGS.out_path += '-{}'.format(
								time.strftime('%Y-%m-%dT%H-%M-%S',time.localtime()))
			train()
			
