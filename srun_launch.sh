PROJ_DIR=$n/space2/LLM-Shearing

echo ${SLURM_NODEID} 
composer --node_rank ${SLURM_NODEID} ./scripts/train/train_sparse.py $@  