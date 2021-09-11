from dataloader import Data_loader
from arg_parser import arg_parser
from utils import *
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import math
import os
from cvxopt import matrix, solvers
try:
	from gurobipy import *
except:
	pass

import itertools

FLAGS = arg_parser().parse_args()


if FLAGS.dataset_name == 'cov':
	class_sizes=np.array([211560,280440])
elif FLAGS.dataset_name == 'mnist':
	class_sizes = np.array([6903, 7877, 6990, 7141, 6824, 6313, 6876, 7293, 6825, 6958])
elif FLAGS.dataset_name == 'cifar10':
	class_sizes = np.array([6000]*10)

def distribute_data():
	try:
		os.makedirs(FLAGS.client_data_path)

		print('distributing data')

		# load all data into data loader
		# reorder data by classes if distributing==class and is_dirichlet==False
		all_tr_data = Data_loader('%s/measures'%(FLAGS.data_path),
			'%s/labels'%(FLAGS.data_path), num_classes=FLAGS.num_classes, shuffle=FLAGS.shuffle,
			distributing=FLAGS.distributing, is_dirichlet=FLAGS.is_dirichlet)
		all_tr_data.reorder_data()
		if FLAGS.shuffle_once:
			all_tr_data.shuffle_once()

		if not FLAGS.is_dirichlet:
			for client_id in range(FLAGS.num_clients):
				client_data = Data_loader('', '', all_tr_data.measures, all_tr_data.labels, 
						partition_id=client_id, total_partition=FLAGS.num_clients, shuffle=FLAGS.shuffle,
						distributing=FLAGS.distributing, is_dirichlet=FLAGS.is_dirichlet)
				client_data.distribute_data()
				client_data.write_to_file(FLAGS.client_data_path,client_id)
		elif not FLAGS.feature_non_iid:
			alphas = np.repeat(FLAGS.alpha, FLAGS.num_clients)
			betas = np.repeat(FLAGS.beta, FLAGS.num_classes)

			client_data_distribution = dirichlet_distribute_data(
				alphas,
				betas,
				class_sizes=class_sizes, 
				q_iterations=500000, 
				burn_in_iterations=500000,
				dirichlet_sample_maxiter=100,
				qp_maxiter=500)
			
			np.savetxt('{}/client_data_distribution.array'.format(FLAGS.client_data_path), client_data_distribution, fmt='%.8f', delimiter=',')

			for client_id in range(FLAGS.num_clients):
				client_data = Data_loader('', '', all_tr_data.measures, all_tr_data.labels, 
						partition_id=client_id, total_partition=FLAGS.num_clients, shuffle=FLAGS.shuffle,
						distributing=FLAGS.distributing, is_dirichlet=FLAGS.is_dirichlet)
				client_data.distribute_data(client_data_distribution)
				client_data.shuffle_once()
				client_data.write_to_file(FLAGS.client_data_path,client_id)
		else:
			dirichlet_distribute_data_feature_class(FLAGS.alpha, FLAGS.beta, FLAGS.theta_t)

	except FileExistsError:
		print('data exist')
		return

def write_combined_data(client_ids, wdir, dataset_type):
	client_data = dict(
		zip(client_ids, 
			[Data_loader(
				'%s/measures_%s'%(FLAGS.client_data_path,client_id), 
				'%s/labels_%s'%(FLAGS.client_data_path,client_id), 
				shuffle=FLAGS.shuffle) 
			for client_id in client_ids]))

	combined_data = Data_loader('', '', 
			client_data[client_ids[0]].measures,
			client_data[client_ids[0]].labels,
			shuffle=FLAGS.shuffle)

	for client_id in client_ids[1:]:
		combined_data.append(client_data[client_id])

	combined_data.write_to_file(wdir,dataset_type)

def combine_data(simple_non_iid=False, sample_sizes=None):
	# random seed for data
	np.random.seed(FLAGS.seed)

	# split clients into training and validation
	client_id_perm = list(range(FLAGS.num_clients))

	if simple_non_iid:
		num_clients_by_class = FLAGS.num_clients * sample_sizes / sample_sizes.sum()
		
		te_client_ids = []
		val_client_ids = []
		tr_client_ids = []
		start = 0
		for c in range(len(sample_sizes)):
			n = int(num_clients_by_class[c])
			end = start + n
			num_te_clients = round(n * FLAGS.client_te_ratio)
			num_val_clients = round(n * FLAGS.client_val_ratio)
			num_tr_clients = FLAGS.num_clients - num_te_clients - num_val_clients
			te_client_ids += client_id_perm[start:end][:num_te_clients]
			val_client_ids += client_id_perm[start:end][num_te_clients:(num_te_clients+num_val_clients)]
			tr_client_ids += client_id_perm[start:end][(num_te_clients+num_val_clients):]
			start = end
	else:
		# Do not shuffle if you want the validation data
		# to be the same as centralized experiment
		np.random.shuffle(client_id_perm)

		num_te_clients = int(FLAGS.num_clients * FLAGS.client_te_ratio)
		num_val_clients = int(FLAGS.num_clients * FLAGS.client_val_ratio)
		num_tr_clients = FLAGS.num_clients - num_te_clients - num_val_clients

		te_client_ids = client_id_perm[:num_te_clients]
		val_client_ids = client_id_perm[num_te_clients:(num_te_clients+num_val_clients)]
		tr_client_ids = client_id_perm[(num_te_clients+num_val_clients):]

	np.savetxt('{}/tr_client_ids.lst'.format(FLAGS.client_data_path), tr_client_ids, fmt='%d', delimiter=',')
	np.savetxt('{}/val_client_ids.lst'.format(FLAGS.client_data_path), val_client_ids, fmt='%d', delimiter=',')
	np.savetxt('{}/te_client_ids.lst'.format(FLAGS.client_data_path), te_client_ids, fmt='%d', delimiter=',')

	with open('{}/split_config.txt'.format(FLAGS.client_data_path),'a') as f:
		print('training clients:', tr_client_ids, file=f)
		print('validation clients:', val_client_ids, file=f)
		print('test clients:', te_client_ids, file=f)

	write_combined_data(te_client_ids, FLAGS.client_data_path, 'test')
	write_combined_data(val_client_ids, FLAGS.client_data_path, 'validation')
	write_combined_data(tr_client_ids, FLAGS.client_data_path, 'train')

def create_folds(n):
	np.random.seed(FLAGS.seed)
	client_id_perm = np.arange(FLAGS.num_clients)
	np.random.shuffle(client_id_perm)
	# kf = KFold(n_splits=int(1.0/FLAGS.client_te_ratio))
	kf = KFold(n_splits=n)
	splits = list(zip(kf.split(client_id_perm)))
	fold0_size = len(splits[0][0][1])
	for i in range(n):
		train_cid = client_id_perm[splits[i][0][0]][fold0_size:]
		val_cid = client_id_perm[splits[i][0][0]][:fold0_size]
		test_cid = client_id_perm[splits[i][0][1]]
		np.savetxt('{}/fold{}_tr_client_ids.lst'.format(FLAGS.client_data_path, i), train_cid, fmt='%d', delimiter=',')
		np.savetxt('{}/fold{}_val_client_ids.lst'.format(FLAGS.client_data_path, i), val_cid, fmt='%d', delimiter=',')
		np.savetxt('{}/fold{}_te_client_ids.lst'.format(FLAGS.client_data_path, i), test_cid, fmt='%d', delimiter=',')


def dirichlet_distribute_data(alphas, betas, class_sizes=np.array([211560,280440]), q_iterations=100, burn_in_iterations=100, dirichlet_sample_maxiter=100, qp_maxiter=100):
	# class non-iid 
	#
	# K: number of classes
	# T: number of clients
	# C: class sizes
	# N: total number of samples
	# n: client sample propotion (samples from dirichlet distribution of clients)
	# ct: class distribution on clients (samples from dirishlet distribution of classes for each client)
	# nN_raw = n*N: number of samples on clients (decimal)
	# nN: rounded number of samples

	np.random.seed(FLAGS.seed)
	T = FLAGS.num_clients
	K = FLAGS.num_classes
	C = class_sizes
	N = sum(C)

	def round_samples_of_clients(maxiter=1000):
		for i in range(maxiter):
			round_nN = np.round(nN_raw).astype(int)

			diff = round_nN.sum() - N
			n_diff = abs(diff)
			T = len(n)
			selected_clients = np.random.choice(range(T), n_diff)
			unique_clients, cnts = np.unique(selected_clients, return_counts=True)
			round_nN[unique_clients] -= np.sign(diff) * cnts

			if (round_nN == 0).sum() == 0:
				return round_nN

		print('Cannot round samples of clients to numbers greater than 0.')

	def round_qp_solution(a, maxiter=1000):
		for i in range(maxiter):
			anew = np.copy(a)
			anew = np.round(anew)
			anew = anew.astype(int)

			for k in range(K):
				diff = anew[:,k].sum() - C[k]
				n_diff = abs(diff)
				T = anew.shape[0]
				selected_clients = np.random.choice(range(T), n_diff)
				unique_clients, cnts = np.unique(selected_clients, return_counts=True)
				anew[unique_clients,k] -= np.sign(diff) * cnts

			if (anew < 0).sum() == 0:
				return anew

		print('Cannot round qp solution to numbers greater than 0.')

	for i in range(dirichlet_sample_maxiter):
		n = np.random.dirichlet(alphas)
		nN_raw = n*N
		if sum(nN_raw < 0.5) == 0:
			ct = np.random.dirichlet(betas,T)
			break

	nN = round_samples_of_clients()
	a_star = ct * np.repeat(nN,K).reshape(T,K)
	a_flatten = a_star.flatten()

	def gurobi_solver():
		m = Model('class_non_iid')
		variables = [m.addVar(vtype=GRB.CONTINUOUS, name='a%d'%(i+1), lb=0) for i in range(T*K)]
		constraints = [m.addConstr(sum(variables[(t*K):((t+1)*K)])==nN[t]) 
							for t in range(T)] + [m.addConstr(sum(variables[k:(T*K):K])==C[k]) 
													for k in range(K-1)]
		m.setObjective(sum((variables[i] - a_flatten[i])*(variables[i] - a_flatten[i]) for i in range(T*K)), GRB.MINIMIZE)
		m.optimize()
		return np.array([v.x for v in m.getVars()]).reshape((T,K))

	def construct_A():
		arr = np.array([np.repeat(1,K)] + [np.repeat(0,K)])
		idx = [1 for _ in range(T)]
		A0 = np.empty((T,0), int)
		for i in range(T):
			idx[i] = 0
			A0 = np.append(A0,arr[idx],axis=1)
			idx[i] = 1

		A1 = np.hstack([np.identity(K) for _ in range(T)])[:-1,:]

		return np.vstack((A0,A1))

	def solve():
		P = matrix(np.identity(K*T))
		q = matrix(-a_star.flatten())

		G = matrix(np.vstack((-1*np.identity(K*T),np.identity(K*T))), tc='d')
		h = matrix(np.hstack((np.repeat(0,K*T),np.repeat(N-1,K*T))), tc='d') 

		A = matrix(construct_A(), tc='d')
		b = matrix(np.append(nN,C[:-1]), tc='d')

		if qp_maxiter is not None:
			solvers.options['maxiters'] = qp_maxiter
		sol = solvers.qp(P,q,G,h,A,b)
		return np.array(sol['x']).reshape(T,K)

	def B(a):
		ii = np.random.choice(range(T), 2)
		jj = np.random.choice(range(K), 2)

		i, j = ii[0], jj[0]
		i_bar, j_bar = ii[1], jj[1]
		e = min(a[i,j], a[i_bar,j_bar])
		if e == 0:
			e_bar = e
		else:
			e_bar = np.random.uniform(0,e*FLAGS.searching_step_pct)

		a[i,j] -= e_bar
		a[i_bar,j_bar] -= e_bar
		a[i,j_bar] += e_bar
		a[i_bar,j] += e_bar

		return a

	def loss(a):
		return np.linalg.norm(a-a_star)

	h = float('inf')
	try:
		qp_a = gurobi_solver()
	except:
		qp_a = solve()
	# qp_a = round_qp_solution(solve())

	opt_loss = loss(qp_a)
	print('loss of qp solution:', opt_loss)

	random_a = np.copy(qp_a)

	for i in range(burn_in_iterations):
		B(random_a)
		random_loss = loss(random_a)
		if random_loss > (1+FLAGS.lower_ratio)*opt_loss and random_loss < (1+FLAGS.upper_ratio)*opt_loss:
			print('random solution within desired range found with %s iters.'%i)
			print(random_loss)
			print('======================')
			break

	h = math.inf
	best_a = np.copy(random_a)

	for i in range(q_iterations):
		B(random_a)
		g = loss(random_a)
		if g < h:
			h = g
			best_a = np.copy(random_a)
			print(h)

	return round_qp_solution(best_a)

def dirichlet_distribute_data_feature_class(mu, lambda_t, theta_t, class_sizes=np.array([211560,280440]), dirichlet_sample_maxiter=100):
	# feature & class non-iid 
	#
	# K: number of classes
	# T: number of clients
	# C: class sizes
	# N: total number of samples
	# n: client sample propotion (samples from dirichlet distribution of clients)
	# ct: class distribution on clients (samples from dirishlet distribution of classes for each client)
	# nN_raw = n*N: number of samples on clients (decimal)
	# nN: rounded number of samples

	def round_samples_of_clients(maxiter=1000):
		for i in range(maxiter):
			round_nN = np.round(nN_raw).astype(int)

			diff = round_nN.sum() - N
			n_diff = abs(diff)
			T = len(n)
			selected_clients = np.random.choice(range(T), n_diff)
			unique_clients, cnts = np.unique(selected_clients, return_counts=True)
			round_nN[unique_clients] -= np.sign(diff) * cnts

			if (round_nN == 0).sum() == 0:
				return round_nN

		print('Cannot round samples of clients to numbers greater than 0.')

	def round_qp_solution(a, maxiter=10000):
		anew = np.copy(a)
		anew = np.round(anew)
		anew = anew.astype(int)

		for k in range(anew.shape[1]):
			diff = anew[:,k].sum() - Cu[k]
			n_diff = abs(diff)
			T = anew.shape[0]

			for i in range(maxiter):
				if np.sign(diff) == -1:
					selected_clients = np.random.choice(range(T), n_diff)
				else:
					selected_clients = np.random.choice(np.where(anew[:,k])[0], n_diff)

				unique_clients, cnts = np.unique(selected_clients, return_counts=True)
				if all(anew[unique_clients,k] - np.sign(diff) * cnts >= 0):
					anew[unique_clients,k] -= np.sign(diff) * cnts
					break

			if i == maxiter:
				print('Cannot round qp solution to numbers greater than 0.')


		if (anew < 0).sum() == 0:
			return anew

	df = pd.read_csv('%s/measures'%(FLAGS.data_path),header=None)
	labels = pd.read_csv('%s/labels'%(FLAGS.data_path),header=None)
	df[df.shape[1]]=np.argmax(labels.values, axis=1)

	tdf = df.loc[:,10:]
	bucket_sizes = [2]*tdf.shape[1]

	buckets, cnts = np.unique(tdf, axis=0, return_counts=True)

	indices = [np.where(np.all(tdf.values==b,axis=1))[0] for b in buckets]

	np.random.seed(FLAGS.seed)
	T = FLAGS.num_clients
	K = FLAGS.num_classes
	P = class_sizes 
	N = sum(P)
	U = buckets
	num_u = len(U)
	Cu = cnts 

	# get the desired feature sampling fjt and desired class sampling lt
	for i in range(dirichlet_sample_maxiter):
		n = np.random.dirichlet(np.repeat(mu, T))
		nN_raw = n*N
		if sum(nN_raw < 0.5) == 0:
			lt = np.random.dirichlet(np.repeat(lambda_t, K),T)
			fjt = [np.random.dirichlet(np.repeat(theta_t, d_j),T) for d_j in bucket_sizes[:-1]]
			break

	nN = round_samples_of_clients()
	a_star_classes = lt * np.repeat(nN,K).reshape(T,K)

	a_star_buckets = [np.empty([T,b]) for b in bucket_sizes[:-1]]
	for j in range(len(bucket_sizes)-1):
		for i in range(bucket_sizes[j]):
			a_star_buckets[j][:,i] = fjt[j][:,i] * nN

	# build QP model and solve
	m = Model('features_classes_non_iid')

	variables = np.reshape(
		[m.addVar(vtype=GRB.CONTINUOUS, name='a%d'%(i+1), lb=0) for i in range(T*num_u)], 
		(T,num_u))

	constraints = [m.addConstr(
					sum(variables[t,:])==nN[t]) for t in range(T)] \
				+ [m.addConstr(
					sum(variables[:,u])==Cu[u]) for u in range(num_u-1)]

	obj = sum(
			sum(variables[t,u]*variables[t,u] for u in np.where(buckets[:,-1] == k)[0])
			+ sum(2*variables[t,p[0]]*variables[t,p[1]] 
				for p in list(itertools.combinations(np.where(buckets[:,-1] == k)[0],2)))
			+ a_star_classes[t,k]*a_star_classes[t,k] 
			- 2*sum(variables[t,u] for u in np.where(buckets[:,-1] == k)[0])*a_star_classes[t,k]
					for t in range(T)
						for k in range(FLAGS.num_classes)) \
		+ sum(
			sum(variables[t,u]*variables[t,u] for u in np.where(buckets[:,j] == i)[0])
			+ sum(2*variables[t,p[0]]*variables[t,p[1]] 
				for p in list(itertools.combinations(np.where(buckets[:,j] == i)[0],2)))
			+ a_star_buckets[j][t,i]*a_star_buckets[j][t,i] 
			- 2*sum(variables[t,u] for u in np.where(buckets[:,j] == i)[0])*a_star_buckets[j][t,i]
					for t in range(T)
						for j in range(len(bucket_sizes)-1)
							for i in range(bucket_sizes[j]))
	m.setObjective(obj, GRB.MINIMIZE)
	m.optimize()

	res = np.array([v.x for v in m.getVars()]).reshape((T,num_u))
	np.save('{}/feature_class_non_iid_qp_raw_res_100clients_all_cat_features_seed1122'.format(FLAGS.client_data_path), res_raw)

	def distribute_sample_index(bucket_indices, client_bucket_distribution):
		"""
		returns a list of sample indices on clients. Each element is a list of indices on a client.

		type buckets: list[list]
		type bucket_indices: list[list]
		type client_bucket_distribution: ndarray, (num_client X num_buckets)
		rtype: list[list]
		"""
		bucket_indices = np.copy(bucket_indices)
		sample_indices = [[] for _ in range(FLAGS.num_clients)]
		for client in range(FLAGS.num_clients):
			for bi in range(len(bucket_indices)):
				nsamples = client_bucket_distribution[client,bi]
				sample_indices[client] += list(bucket_indices[bi][:nsamples])
				bucket_indices[bi] = bucket_indices[bi][nsamples:]
		return sample_indices

	res = round_qp_solution(res)
	sample_index = distribute_sample_index(indices, res)

	for cli in range(FLAGS.num_clients):
		indx = sample_index[cli]
		client_data = df.loc[indx,:]
		np.savetxt(FLAGS.client_data_path + '/measures_%s'%cli, client_data.values[:,:-1], fmt='%d', delimiter=',')
		np.savetxt(FLAGS.client_data_path + '/labels_%s'%cli, client_data.values[:,-1], fmt='%d', delimiter=',')

if __name__ == '__main__':
	if not os.path.isdir(FLAGS.client_data_path):
		distribute_data()
		create_folds(FLAGS.n_eval_folds)
