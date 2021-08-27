Implementation of RADFed in paper Aggregation Delayed Federated Learning

## Datasets

Non-IID datasets used in paper: [COVFEAT](https://www.dropbox.com/s/dfy32fuc8cuqcm4/100_client_data_dirichlet_noniid_cat_features_classes_random_qp_alpha1_lambda0.1_theta0.1_5folds_seed1122.tar.gz?dl=1), [COVCLS](https://www.dropbox.com/s/1kmznrszez0psx0/100_client_data_dirichlet_noniid_classes_random_qp_alpha1_beta0.1_0.1-0.2opt_loss_piter5e5_biter5e5_5folds_seed1020.tar.gz?dl=1), [Cifar10](https://www.dropbox.com/s/rzuvemautwlx8pj/100_client_data_dirichlet_noniid_classes_random_qp_alpha1_beta0.1_0.1-0.2opt_loss_piter5e5_biter5e5_seed2366.tar.gz?dl=1), [Shakespeare](https://www.dropbox.com/s/4m8ihsl18kopfad/143_client_data_seed245.tar.gz?dl=1), [MNIST lambda1](https://www.dropbox.com/s/0k327mg7ssrycqi/100_client_data_dirichlet_noniid_classes_random_qp_alpha1_beta0.1_0.1-0.2opt_loss_search0.002_piter5e5_biter5e5_5folds_seed233.tar.gz?dl=1), [MNIST lambda0.1](https://www.dropbox.com/s/hc4rmxdohppaitz/100_client_data_dirichlet_noniid_classes_random_qp_alpha1_beta1_0.1-0.2opt_loss_search0.002_qiter5e5_biter5e5_seed10.tar.gz?dl=1)

[Mobilenet V2 checkpoint](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz) from its [official github repo](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet)

## Train RADFed

Experiments were performed on Linux with GeForce RTX 2080 Ti (11GB).

Covertype

`python radfed.py --modelname=FFN --score=acc --num_round=1000 --num_clients=100 --num_shuffle_round=22 --client_data_path=data/covtype/100_client_data_dirichlet_noniid_cat_features_classes_random_qp_alpha1_lambda0.1_theta0.1_5folds_seed1122 --out_path=out/test_run --num_classes=2 --feature_size=49 --save_file --num_tr_workers=2 --num_gpus=0 --gpu_ids=0 --learning_rate=0.2 --batch_size=256`


MNIST

`python radfed.py --modelname=FFN --score=acc --num_round=30 --num_clients=100 --num_shuffle_round=15 --client_data_path=data/mnist/100_client_data_dirichlet_noniid_classes_random_qp_alpha1_beta0.1_0.1-0.2opt_loss_search0.002_piter5e5_biter5e5_5folds_seed233 --out_path=out/test_run --num_classes=10 --feature_size=784 --save_file --num_tr_workers=2 --num_gpus=0 --gpu_ids=0 --learning_rate=0.002 --batch_size=10 --num_local_epochs=20`

Cifar10

`python radfed.py --modelname=mbnt --score=acc --num_round=100 --num_clients=100 --num_shuffle_round=15 --client_data_path=data/cifar10/100_client_data_dirichlet_noniid_classes_random_qp_alpha1_beta0.1_0.1-0.2opt_loss_piter5e5_biter5e5_seed2366 --out_path=out/test_run --num_classes=10 --save_file --num_tr_workers=2 --num_gpus=1 --gpu_ids=0 --learning_rate=0.2 --batch_size=16`

Shakespeare

`python radfed.py --modelname=lstm --score=acc --num_round=100 --num_clients=143 --num_shuffle_round=15 --client_data_path=data/shakespeare/143_client_data_seed245 --out_path=out/test_run --num_classes=80 --save_file --num_tr_workers=2 --num_gpus=1 --gpu_ids=0 --learning_rate=0.5 --batch_size=256`


## Train RADFed-IS

Set `--return_grad_norm=2` and `--ISalpha=<IS_alpha>`


## Generate the non-IID data above from original datasets

Covertype: download [raw data](https://www.dropbox.com/sh/3ca3j7hz6r09d7v/AABDA3x_25T9qrZ_XozDu_dLa?dl=0), then run

COVCLS:
`python distribute_data.py --data_path=<path_to_measures> --client_data_path=<dist_path> --num_clients=100  --distributing=class --client_te_ratio=0.1 --client_val_ratio=0.1 --is_dirichlet --num_classes=2 --seed=1020 --alpha=1 --beta=0.1 --searching_step_pct=0.0002 --lower_ratio=0.1 --upper_ratio=0.2 --n_eval_folds=5`

COVFEAT:
`python distribute_data.py --data_path=<path_to_measures> --client_data_path=<dist_path> --num_clients=100 --feature_non_iid --distributing=class --client_te_ratio=0.1 --client_val_ratio=0.1 --is_dirichlet --num_classes=2 --seed=1122 --alpha=1 --beta=0.1 --theta_t=0.1 --searching_step_pct=0.0002 --lower_ratio=0.1 --upper_ratio=0.2 --n_eval_folds=5`

MNIST: download [raw data](https://www.dropbox.com/sh/lfo460jc6yeycyb/AAAOR4DgzqUz9LASyyMFRfkka?dl=0), then run

MNIST-1:
`python distribute_data.py --data_path=<path_to_measures> --client_data_path=<dist_path> --num_clients=100  --distributing=class --client_te_ratio=0.1 --client_val_ratio=0.1 --is_dirichlet --num_classes=10 --seed=10 --alpha=1 --beta=1 --searching_step_pct=0.0002 --lower_ratio=0.1 --upper_ratio=0.2 --n_eval_folds=5`

MNIST-01:
`python distribute_data.py --data_path=<path_to_measures> --client_data_path=<dist_path> --num_clients=100  --distributing=class --client_te_ratio=0.1 --client_val_ratio=0.1 --is_dirichlet --num_classes=10 --seed=233 --alpha=1 --beta=0.1 --searching_step_pct=0.0002 --lower_ratio=0.1 --upper_ratio=0.2 --n_eval_folds=5`

Cifar10: download [raw data](https://www.dropbox.com/sh/ydjzhrkj6kl08tp/AAAD1zX8kxK-_fL1FNhfyxOqa?dl=0), then run

`python distribute_data.py --data_path=<path_to_measures> --client_data_path=<dist_path> --num_clients=100  --distributing=class --client_te_ratio=0.1 --client_val_ratio=0.1 --is_dirichlet --num_classes=10 --seed=2366 --alpha=1 --beta=0.1 --searching_step_pct=0.0002 --lower_ratio=0.1 --upper_ratio=0.2 --n_eval_folds=5`

Shakespeare: download [raw data & scripts](https://www.dropbox.com/sh/88cw9h1mp1rafik/AACNGcrZZ0ODqY-OqmmFNjAma?dl=0), then run preprocess.ipynb


