CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 composer train/train.py \
  train/yamls/pretrain/llama2-7b.yaml \
  save_folder=llama2-7b-20kcali-2x-finetune-15000batch_lr_1e-5_2048_squarehead