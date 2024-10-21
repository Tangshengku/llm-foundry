CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 composer train/train_sparse.py \
  train/yamls/pretrain/llama3-8b.yaml \
  save_folder=llama3-8b