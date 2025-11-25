#!/bin/bash

python src/pibnet/train.py \
dataset_train='datasets/helmholtz_dirichlet' \
dataset_val='datasets/helmholtz_dirichlet_test' \
dataset_test='datasets/helmholtz_dirichlet_test' \
compare_normalized=True \
predict_normalized=True \
data.distant_edge_sample_period=2

python src/pibnet/train.py \
dataset_train='datasets/helmholtz_neumann' \
dataset_val='datasets/helmholtz_neumann_test' \
dataset_test='datasets/helmholtz_neumann_test' \
compare_normalized=False \
predict_normalized=False \
data.distant_edge_sample_period=2

python src/pibnet/train.py \
dataset_train='datasets/laplace_dirichlet' \
dataset_val='datasets/laplace_dirichlet_test' \
dataset_test='datasets/laplace_dirichlet_test' \
compare_normalized=True \
predict_normalized=True \
data.distant_edge_sample_period=2