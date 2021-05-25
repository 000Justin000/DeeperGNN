#!/bin/sh

GPU=0

# echo "=====Cora====="
# CUDA_VISIBLE_DEVICES=${GPU} python dagnn.py --dataset=Cora --weight_decay=0.005 --K=10 --dropout=0.8 --random_splits=True

# echo "=====CiteSeer====="
# CUDA_VISIBLE_DEVICES=${GPU} python dagnn.py --dataset=CiteSeer --weight_decay=0.02 --K=10 --dropout=0.5 --random_splits=True

# echo "=====PubMed====="
# CUDA_VISIBLE_DEVICES=${GPU} python dagnn.py --dataset=PubMed --weight_decay=0.005 --K=20 --dropout=0.8 --random_splits=True 

# echo "=====Coauthor CS====="
# CUDA_VISIBLE_DEVICES=${GPU} python dagnn.py --dataset=cs --weight_decay=0 --K=5 --dropout=0.8 

# echo "=====Coauthor Physics====="
# CUDA_VISIBLE_DEVICES=${GPU} python dagnn.py --dataset=physics --weight_decay=0 --K=5 --dropout=0.8 

# echo "=====Ising+====="
# CUDA_VISIBLE_DEVICES=${GPU} python dagnn.py --dataset=ising+

# echo "=====Ising-====="
# CUDA_VISIBLE_DEVICES=${GPU} python dagnn.py --dataset=ising-

# echo "=====MRF+====="
# CUDA_VISIBLE_DEVICES=${GPU} python dagnn.py --dataset=mrf+

# echo "=====MRF-====="
# CUDA_VISIBLE_DEVICES=${GPU} python dagnn.py --dataset=mrf-

# echo "=====County_Facebook====="
# CUDA_VISIBLE_DEVICES=${GPU} python dagnn.py --dataset=county_facebook

# echo "=====Sexual_Interaction====="
# CUDA_VISIBLE_DEVICES=${GPU} python dagnn.py --dataset=sexual_interaction
