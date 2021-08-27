import ray
import numpy as np
import os

def get_test_scores(test_workers, weights, score, average='weighted'):
	test_worker_scores_losses = [test_worker.compute_scores_losses.remote(weights, score=score, average=average)
				for test_worker in test_workers]

	testwscoreslosses = [client_test_scores_losses 
					for testwscoreloss in test_worker_scores_losses 
					for client_test_scores_losses in ray.get(testwscoreloss)]

	test_scores = [client_test_scores_losses[0] for client_test_scores_losses in testwscoreslosses]
	test_losses = [client_test_scores_losses[1] for client_test_scores_losses in testwscoreslosses]

	return np.mean(test_scores), np.mean(test_losses)


def test_model(test_workers, FLAGS):
	best_weights = np.loadtxt(FLAGS.out_path+'/best_weights.ndarray', dtype=np.float32, delimiter=',')

	f1_micro, _ = get_test_scores(test_workers, best_weights, 'F1', 'micro')
	f1_macro, _ = get_test_scores(test_workers, best_weights, 'F1', 'macro')
	f1_weighted, _ = get_test_scores(test_workers, best_weights, 'F1', 'weighted')
	f1_binary, _ = get_test_scores(test_workers, best_weights, 'F1', 'binary')
	acc, _ = get_test_scores(test_workers, best_weights, 'acc')
	auc, _ = get_test_scores(test_workers, best_weights, 'auc')
	
	with open('{}/results_more_score.txt'.format(FLAGS.out_path),'w') as f:
		print('Avg test F1 micro: {:.8f}'.format(f1_micro), file=f)
		print('Avg test F1 macro: {:.8f}'.format(f1_macro), file=f)
		print('Avg test F1 weighted: {:.8f}'.format(f1_weighted), file=f)
		print('Avg test F1 binary: {:.8f}'.format(f1_binary), file=f)
		print('Avg test acc: {:.8f}'.format(acc), file=f)
		print('Avg test auc: {:.8f}'.format(auc), file=f)
