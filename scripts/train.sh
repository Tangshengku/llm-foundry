CUDA_VISIBLE_DEVICES=6,7 composer train/train.py \
  train/yamls/pretrain/llama2-7b.yaml \
  save_folder=llama2-7b-20kcali-1.5x-finetune-30000batch_lr_1e-4_4096_fineweb_test