CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 composer train/train_sparse.py \
  /nfs/scistore19/alistgrp/stang/llm-foundry/scripts/train/yamls/pretrain/shearedllama.yaml \
  save_folder=shearedllama_2.7B_fineweb_20B\