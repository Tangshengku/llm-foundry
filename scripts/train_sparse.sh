CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 composer train/train_sparse.py \
  train/yamls/pretrain/llama2-7b.yaml \
  save_folder=evo_search_1.5_0.2B