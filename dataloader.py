import numpy as np
import time
import math
import tensorflow as tf

try:
	from torch.utils.data import Dataset, DataLoader
	import torch
	import random

	def cycle(iterable):
	    while True:
	        for x in iterable:
	            yield x

	class cifar10_dataset(Dataset):
	    def __init__(self, imgs, labels, transforms):
	        self.imgs = imgs.reshape(-1,3,32,32).transpose((0,2,3,1))

	        self.labels = labels
	        self.transform = transforms
	        self.N = imgs.shape[0]

	    def __len__(self):
	        return self.N

	    def __getitem__(self, idx):
	        img = self.imgs[idx]
	        img = self.transform(img).numpy()
	        return {'img': img, 'label': self.labels[idx]}

	def cifar10_dataloader(imgs, labels, transforms, batch_size, n_workers=0, shuffle=False):
	    dataset = cifar10_dataset(imgs, labels, transforms)
	    dl = DataLoader(dataset, num_workers=n_workers, batch_size=batch_size, shuffle=shuffle,
	                             collate_fn=cifar10_collate_fn, drop_last=False, pin_memory=False)
	        
	    return dl

	def cifar10_collate_fn(data):
	    collate_images = []
	    collate_labels = []
	    for d in data:
	        collate_images.append(d['img'])
	        collate_labels.append((d['label']))
	    collate_images = np.stack(collate_images, axis=0).transpose((0,2,3,1))
	    collate_labels = np.stack(collate_labels, axis=0)

	    return {
	        'imgs': collate_images,
	        'labels': collate_labels
	    }

except:
	pass

class Data_loader():
	"""Loading data from given input file name and label file name"""

	def __init__(self, measure_fn, label_fn, measures=None, labels=None, num_classes=2, dtype=np.float32, delimiter=',', 
		features=None, transforms=None, partition_id=None, total_partition=None, 
		shuffle=False, distributing='iid', is_dirichlet=False, imputation=None, seed=8888,
		one_hot=True):

		np.random.seed(seed)

		self.shuffle = shuffle

		if measure_fn == '':
			self.measures = measures
			self.labels = labels
		else:
			self.measures = np.loadtxt(measure_fn,dtype=dtype,delimiter=delimiter)
			self.labels = np.loadtxt(label_fn,dtype=dtype,delimiter=delimiter)
		
		if imputation:
			self.impute(imputation)

		self.num_examples = self.measures.shape[0]

		self.partition_id = partition_id
		self.total_partition = total_partition
		self.distributing = distributing
		self.is_dirichlet = is_dirichlet
		self.transforms = transforms

		if features is not None:
			self.measures = self.measures[:,features]
		
		if one_hot:
			if len(self.labels.shape) < 2:
				self.one_hot_labels = np.zeros((self.num_examples, num_classes)) #np.zeros((self.num_examples, 2))#
				self.one_hot_labels[np.arange(self.num_examples), self.labels.astype(int)] = 1
				self.labels = self.one_hot_labels
			
			self.labels_num = self.labels.argmax(axis=1).astype(int)

			self.class_ids = np.unique(self.labels_num)
		else:
			self.labels_num = self.labels
			self.class_ids = np.unique(self.labels_num)

		self.pos = 0

	def next_batch(self, batch_size, shuffle=False, round_robin=True):
		if batch_size == -1:
			# get full batch
			return self.measures, self.labels

		start = self.pos

		if start + batch_size > self.num_examples:
			# get the rest examples
			rest_num_examples = self.num_examples - start
			rest_measures = self.measures[start:self.num_examples,]
			rest_labels = self.labels[start:self.num_examples,]
			
			if not round_robin:
				self.pos = 0
				batch_measures, batch_labels = rest_measures, rest_labels
			else:
				# shuffle data
				if shuffle or self.shuffle:
					perm = np.arange(self.num_examples)
					np.random.shuffle(perm)
					if self.measures.shape[0] < self.num_examples:
						import sys
						print(self.measures.shape)
						print(self.num_examples)
						sys.exit('here')
					self.measures = self.measures[perm,]
					self.labels = self.labels[perm,]

				# start next epoch
				start = 0
				self.pos = batch_size - rest_num_examples
				end = self.pos
				new_measures = self.measures[start:end,]
				new_labels = self.labels[start:end,]

				batch_measures, batch_labels = np.concatenate((rest_measures, new_measures), axis=0), np.concatenate((rest_labels, new_labels), axis=0)
		else:
			self.pos += batch_size
			end = self.pos
			self.pos %= self.num_examples
			batch_measures, batch_labels = self.measures[start:end], self.labels[start:end]

		return batch_measures, batch_labels

	def rewind(self):
		self.pos = 0

	def shuffle_once(self):
		perm = np.arange(self.num_examples)
		np.random.shuffle(perm)
		self.measures = self.measures[perm,]
		self.labels = self.labels[perm,]

	def random_sample(self):
		idx = np.random.randint(self.num_examples)
		return self.measures[idx:(idx+1)], self.labels[idx:(idx+1)]

	def normalize(self, col_mean=None, col_std=None, cols=None):
		if col_mean is None:
			col_mean = self.measures.mean(axis=0)
		if col_std is None:
			col_std = self.measures.std(axis=0)
		
		col_std[col_std==0]=1
		
		if cols is not None:
			normed = (self.measures[:,cols] - col_mean[cols]) / col_std[cols]
			self.measures = np.concatenate((normed, self.measures[:,~cols]),axis=1)
		else:
			self.measures = (self.measures - col_mean) / col_std

		return col_mean, col_std

	def minmax(self, col_min=None, col_max=None):
		if col_min is None:
			col_min = self.measures.min(axis=0)
		if col_max is None:
			col_max = self.measures.max(axis=0)

		scaler = col_max - col_min
		scaler[scaler==0] = 1

		self.measures = (self.measures - col_min) / scaler
		return col_min, col_max

	def impute(self, imputation, col_mean=None):
		if imputation == 'mean':
			if col_mean is None:
				col_mean = np.nanmean(self.measures, axis=0)
			inds = np.where(np.isnan(self.measures))
			self.measures[inds] = np.take(col_mean, inds[1])

		return col_mean

	def append(self, other_data_loader):
		self.measures = np.concatenate((self.measures, other_data_loader.measures), axis=0)
		self.labels = np.concatenate((self.labels, other_data_loader.labels), axis=0)
		self.num_examples += other_data_loader.num_examples

	def write_to_file(self, path, suffix):
		np.savetxt('%s/measures_%s'%(path,suffix), self.measures, fmt='%d', delimiter=',')
		np.savetxt('%s/labels_%s'%(path,suffix), self.labels, fmt='%d', delimiter=',')

	def reorder_data(self):
		if self.distributing == 'class':
			self.measures = np.concatenate(
				[self.measures[self.labels_num==c] for c in self.class_ids], axis=0)
			self.labels = np.concatenate(
				[self.labels[self.labels_num==c] for c in self.class_ids], axis=0)
			
	def distribute_data(self, client_data_distribution=None):
		if self.total_partition is None:
			import sys
			sys.exit('Total number of partition is not specified.')
		
		if client_data_distribution is not None:
			client_measures_per_class = []
			client_labels_per_class = []

			class_offset = 0
			for c in self.class_ids:
				start = client_data_distribution[:self.partition_id,c].sum() + class_offset
				end = start + client_data_distribution[self.partition_id,c]
				
				client_measures_per_class.append(self.measures[start:end,])
				client_labels_per_class.append(self.labels[start:end,])
				class_offset += sum(self.labels_num==c)

			self.measures = np.concatenate(client_measures_per_class, axis=0)
			self.labels = np.concatenate(client_labels_per_class, axis=0)
			self.num_examples = self.measures.shape[0]

		elif self.distributing == 'iid' or (self.distributing == 'class' and not self.is_dirichlet):
			# split data into 'total_partition' number of chunks
			# and load one of them specified by 'partition_id'
			start = int(self.partition_id * self.num_examples / self.total_partition)
			end = int((self.partition_id + 1) * self.num_examples / self.total_partition)
			self.measures = self.measures[start:end,]
			self.labels = self.labels[start:end,]
			self.num_examples = self.measures.shape[0]
		
