#!/bin/bash

# Laplace Dirichlet dataset

# 3-obstacle test
python src/pibnet/test.py \
dataset_test='datasets/laplace_dirichlet_test' \
data.candidate_edge_ratio=2 \
checkpoint_reference='wandb_checkpoint_reference' \
data.distant_edge_sample_period=2

# 6-obstacle test
python src/pibnet/test.py \
dataset_test='datasets/laplace_dirichlet_test_6obs' \
data.candidate_edge_ratio=3 \
checkpoint_reference='wandb_checkpoint_reference' \
data.distant_edge_sample_period=2

# 9-obstacle test
python src/pibnet/test.py \
dataset_test='datasets/laplace_dirichlet_test_9obs' \
data.candidate_edge_ratio=3 \
checkpoint_reference='wandb_checkpoint_reference' \
data.distant_edge_sample_period=2

#---------------------------------------------------------------------------------------------
# Helmholtz Dirichlet dataset

# 3-obstacle test
python src/pibnet/test.py \
dataset_test='datasets/helmholtz_dirichlet_test' \
data.candidate_edge_ratio=2 \
checkpoint_reference='wandb_checkpoint_reference' \
data.distant_edge_sample_period=2

# 6-obstacle test
python src/pibnet/test.py \
dataset_test='datasets/helmholtz_dirichlet_test_6obs' \
data.candidate_edge_ratio=3 \
checkpoint_reference='wandb_checkpoint_reference' \
data.distant_edge_sample_period=2

# 9-obstacle test
python src/pibnet/test.py \
dataset_test='datasets/helmholtz_dirichlet_test_9obs' \
data.candidate_edge_ratio=3 \
checkpoint_reference='wandb_checkpoint_reference' \
data.distant_edge_sample_period=2

#---------------------------------------------------------------------------------------------
# Helmholtz Neumann dataset

# 3-obstacle test
python src/pibnet/test.py \
dataset_test='datasets/helmholtz_neumann_test' \
data.candidate_edge_ratio=2 \
checkpoint_reference='wandb_checkpoint_reference' \
data.distant_edge_sample_period=2

# 6-obstacle test
python src/pibnet/test.py \
dataset_test='datasets/helmholtz_neumann_test_6obs' \
data.candidate_edge_ratio=3 \
checkpoint_reference='wandb_checkpoint_reference' \
data.distant_edge_sample_period=2

# 9-obstacle test
python src/pibnet/test.py \
dataset_test='datasets/helmholtz_neumann_test_9obs' \
data.candidate_edge_ratio=3 \
checkpoint_reference='wandb_checkpoint_reference' \
data.distant_edge_sample_period=2