#!/bin/bash

save_name='laplace_dirichlet'

# Generation of the train set with 3 obstacles
python src/data_generation/data_gen.py \
num_samples=10000 \
save_name=$save_name \
seed=42 \
equation='laplace'

# Generation of the test set with 3 obstacles
python src/data_generation/data_gen.py \
num_samples=1000 \
save_name=$save_name'_test' \
seed=42000000 \
equation='laplace'

# Generation of the test set with 6 obstacles
python src/data_generation/data_gen.py \
num_samples=1000 \
save_name=$save_name'_test_6obs' \
seed=42000000 \
equation='laplace' \
obstacles.number.min=6 \
obstacles.number.max=7

# Generation of the test set with 9 obstacles
python src/data_generation/data_gen.py \
num_samples=1000 \
save_name=$save_name'_test_9obs' \
seed=42000000 \
equation='laplace' \
obstacles.number.min=9 \
obstacles.number.max=10

# Creation of multi-level sets

python src/data_generation/to_multi_level_binary_tree.py \
save_name=$save_name

python src/data_generation/to_multi_level_binary_tree.py \
save_name=$save_name'_test'

python src/data_generation/to_multi_level_binary_tree.py \
save_name=$save_name'_test_6obs'

python src/data_generation/to_multi_level_binary_tree.py \
save_name=$save_name'_test_9obs'
