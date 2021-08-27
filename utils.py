import matplotlib.pyplot as plt
import pickle
import os
import shutil
import re
import numpy as np

class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'

def mkdir(directory):
	try:
		os.makedirs(directory)
	except FileExistsError:
		pass

def dump(item, fn):
	with open(fn, 'wb') as fp:
		pickle.dump(item, fp)

def load(fn):
	with open(fn, 'rb') as fp:
		return pickle.load(fp)

def plot(x, y1, y2, labels, fn, title='', xlabel='epochs'):
	fig, ax = plt.subplots()
	ax.set_title(title)
	ax.plot(x, y1, label=labels[0])
	ax.plot(x, y2, label=labels[1])
	ax.set_xlabel(xlabel)
	ax.legend()
	fig.savefig(fn)
	plt.close(fig)

def plot_1(x, y, label, fn, title='', xlabel='epochs'):
	fig, ax = plt.subplots()
	ax.set_title(title)
	ax.plot(x, y, label=label)
	ax.set_xlabel(xlabel)
	ax.legend()
	fig.savefig(fn)
	plt.close(fig)

def copydir(src, dst, symlinks=False, ignore=None):
	if not os.path.exists(dst):
		os.makedirs(dst)
	for item in os.listdir(src):
		s = os.path.join(src, item)
		d = os.path.join(dst, item)
		if os.path.isdir(s):
			copydir(s, d, symlinks, ignore)
		else:
			if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
				shutil.copy2(s, d)

def wilcoxon_test(fns, folds, seeds, metric='F1', alternative='two-sided', printing=False):
	from scipy.stats import wilcoxon
	
	res1 = [get_score(fns[0]%(fold,seed), metric) for fold in folds for seed in seeds]
	res2 = [get_score(fns[1]%(fold,seed), metric) for fold in folds for seed in seeds]

	if printing:
		print(res1)
		print(res2)

	return wilcoxon(res1, res2, alternative=alternative)

def get_score(fn, metric, test=True):
	try:
		with open(os.path.join(fn,'results.txt'),'r') as f:
			t = f.read()
		
		if test:
			return float(re.findall(r'Avg test %s: (\d+\.\d+)'%metric, t)[0])

		return float(re.findall(r'Best Avg validation %s: (\d+\.\d+)'%metric, t)[0])
	except:
		return 0

def get_more_score(fn, metric, average=None):
	try:
		with open(os.path.join(fn,'results_more_score.txt'),'r') as f:
			t = f.read()
		
		if average:
			return float(re.findall(r'Avg test %s %s: (\d+\.\d+)'%(metric, average), t)[0])
		return float(re.findall(r'Avg test %s: (\d+\.\d+)'%metric, t)[0])

	except:
		return 0

def get_num_best_rounds(fn, metric, shr=0):
	try:
		with open(os.path.join(fn,'results.txt'),'r') as f:
			t = f.read()
		r = int(re.findall(r'Best round by %s: (\d+)\.\d+'%metric, t)[0])
		return r if shr == 0 else (r+1)*shr
	except:
		return 0

def get_num_rounds(fn, shr=0):
	try:
		r = len(np.loadtxt(os.path.join(fn, 'avg_training_loss_history.lst')))
		return r if shr == 0 else r*shr
	except:
		return 0

