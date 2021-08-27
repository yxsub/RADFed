import argparse

def arg_parser():
	parser = argparse.ArgumentParser(
		description='Run the synchronous parameter server example.')

	# Classification model settings
	parser.add_argument(
		'--graph_level_seed', 
		default=1, 
		type=int,
		help='Tensorflow graph level seed.')
	parser.add_argument(
		'--seed', 
		default=1234, 
		type=int,
		help='Numpy seed.')
	parser.add_argument(
		'--feature_size', 
		default=54, 
		type=int,
		help='Number of features in the dataset.')
	parser.add_argument(
		'--num_classes', 
		default=3, 
		type=int,
		help='Number of classes in the dataset.')
	parser.add_argument(
		'--hidden_size', 
		default=64, 
		type=int,
		help='Size of hidden layers.')
	parser.add_argument(
		'--optimizer', 
		default='sgd', 
		type=str,
		help='Optimizer.')
	parser.add_argument(
		'--modelname', 
		default='FFN', 
		type=str,
		help='Classifier name.')
	

	# Data settings
	parser.add_argument(
		'--n_eval_folds', 
		default=5, 
		type=int,
		help='Total number of folds used for evaluation.')
	parser.add_argument(
		'--data_path', 
		default='data/', 
		type=str, 
		help='Validation and test data path')
	parser.add_argument(
		'--data_prefix', 
		default='covtype3', 
		type=str, 
		help='Data prefix')
	parser.add_argument(
		'--shuffle', 
		action='store_true',
		help='True if shuffling data in each epoch.')
	parser.add_argument(
		'--shuffle_once', 
		action='store_true',
		help='True if shuffling data before training.')
	parser.add_argument(
		'--round_robin', 
		action='store_true',
		help='True if visiting data in round robin fashion.')
	parser.add_argument(
		'--out_path', 
		default='out/', 
		type=str, 
		help='Output Path')
	parser.add_argument(
		'--move_out_path', 
		default=None, 
		type=str, 
		help='Destination to move out_path to')
	parser.add_argument(
		'--out_suffix', 
		default='', 
		type=str, 
		help='Suffix of output Path')
	parser.add_argument(
		'--resume_path', 
		default='out/', 
		type=str, 
		help='Path of previous trained model to be resumed')

	# Distributing data settings
	parser.add_argument(
		'--dataset_name', 
		default='cov', 
		type=str, 
		help='The name of dataset.')
	parser.add_argument(
		'--distributing', 
		default='iid', 
		type=str, 
		help='The way to distribute data on clients.')
	parser.add_argument(
		'--is_dirichlet', 
		action='store_true', 
		help='Distribute data using Dirichlet.')
	parser.add_argument(
		'--feature_non_iid', 
		action='store_true', 
		help='Distribute non iid features using Dirichlet.')
	parser.add_argument(
		'--searching_step_pct', 
		default=0.1, 
		type=float,
		help='The searching step in finding random solution.')
	parser.add_argument(
		'--lower_ratio', 
		default=0.1, 
		type=float,
		help='Lower ratio of the optimal loss in burn-in period.')
	parser.add_argument(
		'--upper_ratio', 
		default=0.2, 
		type=float,
		help='Upper ratio of the optimal loss in burn-in period.')
	parser.add_argument(
		'--alpha', 
		default=1, 
		type=float,
		help='Dirichlet prior non-IID sizes.')
	parser.add_argument(
		'--beta', 
		default=1, 
		type=float,
		help='Dirichlet prior for non-IID classes.')
	parser.add_argument(
		'--theta_t', 
		default=0.1, 
		type=float,
		help='Dirichlet prior for non-IID features.')
	

	# General training settings
	parser.add_argument(
		'--learning_rate', 
		default=1e-3, 
		type=float,
		help='The learning rate of the model.')
	parser.add_argument(
		'--batch_size', 
		default=256, 
		type=int,
		help='The batch size in optimization.')
	parser.add_argument(
		'--val_batch_size', 
		default=256, 
		type=int,
		help='The batch size in evaluation.')
	parser.add_argument(
		'--fold', 
		default=0, 
		type=int,
		help='The fold id used for evaluation.')
	parser.add_argument(
		'--inner_fold', 
		default=None, 
		type=int,
		help='The fold id used for parameter tuning.')
	parser.add_argument(
		'--update_frequency_ratio', 
		default=0.2, 
		type=float,
		help='The number of inner updates in SVRG is equal to '
		'update_frequency_ratio * number of samples.')
	parser.add_argument(
		'--use_adam', 
		action='store_true',
		help='True if use Adam optimization in SVRG.')
	parser.add_argument(
		'--num_gpus',
		default=0,
		type=float,
		help='Number of GPUs to use for training.')
	parser.add_argument(
		'--gpu_ids',
		default=[1],
		nargs='+',
		type=int,
		help='The ids of GPUs to be used.')
	parser.add_argument(
		'--timeit', 
		action='store_true',
		help='True if report running time.')
	parser.add_argument(
		'--redis_address', 
		default=None, 
		type=str,
		help='The Redis address of the cluster.')
	parser.add_argument(
		'--re_init_client_freq', 
		default=500, 
		type=int,
		help='frequency of re-initting client graphs.')
	parser.add_argument(
		'--window_size', 
		default=100, 
		type=int,
		help='Early stopping - window size (epochs) used to calculate relative F1 increase.')
	parser.add_argument(
		'--threshold', 
		default=0.001, 
		type=float,
		help='Early stopping - threshold of the relative F1 increase.')
	parser.add_argument(
		'--converge_window', 
		default=100, 
		type=int,
		help='Early stopping - the minimal number of epochs when F1 increase is below the threshold.')
	parser.add_argument(
		'--score', 
		default='F1', 
		type=str,
		help='Metrics.')
	parser.add_argument(
		'--n_data_workers', 
		default=0, 
		type=int,
		help='num_workers of torch dataloader')
	parser.add_argument(
		'--save_divergence',
		action='store_true',
		help='calculate and save parameter divergence')
	

	# Experimental settings
	parser.add_argument(
		'--log_device_placement',
		action='store_true',
		help='True if let tf log device placement.')
	parser.add_argument(
		'--save_file', 
		action='store_true',
		help='True if to save files.')
	parser.add_argument(
		'--eval_freq', 
		default=10, 
		type=int,
		help='frequency of evaluating model performance (every "eval_freq" rounds).')
	parser.add_argument(
		'--write_freq', 
		default=10, 
		type=int,
		help='frequency of writing results to file (every "write_freq" rounds).')
	parser.add_argument(
		'--copy_file_freq', 
		default=30000, 
		type=int,
		help='frequency of copying results to file.')
	parser.add_argument(
		'--resume', 
		action='store_true',
		help='True if start training from where it left by given output path.')
	parser.add_argument(
		'--start_round', 
		default=-1, 
		type=int,
		help='The round number from which training resumes.')
	parser.add_argument(
		'--global_normalizing', 
		action='store_true',
		help='True if clients normalize data with global statistics.')
	parser.add_argument(
		'--normalizing', 
		action='store_true',
		help='True if clients normalize data locally.')
	parser.add_argument(
		'--num_normalized_features', 
		default=None, 
		type=int,
		help='Indicates which features to normalize. '
		'The first "num_normalized_features" will be normalized')
	parser.add_argument(
		'--stop_round', 
		default=0, 
		type=int,
		help='Stopping rounds.')
	parser.add_argument(
		'--grad_norm_on_global_weights', 
		action='store_true',
		help='True if clients calculate gradient norm on last global weights.')
	parser.add_argument(
		'--imputation',
		default=None,
		type=str,
		help='imputation method name' )
	parser.add_argument(
		'--numerical_only',
		action='store_true',
		help='True if only use numerical features' )

	# Federated training settings
	parser.add_argument(
		'--num_round', 
		default=200, 
		type=int,
		help='The number of total training rounds.')
	parser.add_argument(
		'--num_local_epochs', 
		default=1, 
		type=int,
		help='The number of local epochs.')
	parser.add_argument(
		'--C',
		default=0.1,
		type=float,
		help='Fraction of clients that perform computation on each round.')
	parser.add_argument(
		'--num_workers', 
		default=4, 
		type=int,
		help='The number of workers to use.')
	parser.add_argument(
		'--num_tr_workers', 
		default=4, 
		type=int,
		help='The number of workers to train.')
	parser.add_argument(
		'--num_val_workers', 
		default=1, 
		type=int,
		help='The number of workers to validate.')
	parser.add_argument(
		'--num_te_workers', 
		default=1, 
		type=int,
		help='The number of workers to test.')
	parser.add_argument(
		'--num_clients', 
		default=4, 
		type=int,
		help='The number clients.')
	parser.add_argument(
		'--client_te_ratio',
		default=0.1,
		type=float,
		help='Ratio of number of test clients.')
	parser.add_argument(
		'--client_val_ratio',
		default=0.1,
		type=float,
		help='Ratio of number of validation clients.')
	parser.add_argument(
		'--client_data_path',
		default='',
		help='Path of distributed client data.')
	parser.add_argument(
		'--save_weights_freq', 
		default=4000, 
		type=int,
		help='frequency of saving best weights.')
	parser.add_argument(
		'--test', 
		action='store_true',
		help='True if only to evaluate the trained model')
	
	
	# RADFed settings
	parser.add_argument(
		'--num_shuffle_round', 
		default=5, 
		type=int,
		help='The number shuffleing rounds.')
	parser.add_argument(
		'--return_grad_norm', 
		default=None,
		type=int,
		help='0-5 if selecting client based on grad norm option 0-5')
	parser.add_argument(
		'--n_mini_batch_grad_norm', 
		default=None,
		type=int,
		help='hyperparameter for return_grad_norm option 3,4&5')
	parser.add_argument(
		'--apply_weights_freq', 
		default=1, 
		type=int,
		help='apply weights every apply_weights_freq rounds')
	parser.add_argument(
		'--ISalpha', 
		default=None, 
		type=float,
		help='alpha in importance sampling')

	# MobileNet experimental setttings
	parser.add_argument(
		'--learning_rate_decay_factor', 
		default=10, 
		type=float,
		help='decay learning rate by this number')
	parser.add_argument(
		'--learning_rate_decay_freq', 
		default=None, 
		type=int,
		help='decay learning rate each'
		'learning_rate_decay_freq number of rounds')


	return parser