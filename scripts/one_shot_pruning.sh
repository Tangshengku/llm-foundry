calibration_data_size=2048
target=2.5
is_prune=true
run_name=evo_search_2.5_from_2_reg
global_train_batch_size=1

# In some cases, bf16 may cause Non-Full rank during inverting the hessian matrix   
use_flash_attn=true

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python train/train.py \
  train/yamls/pretrain/llama2-7b.yaml \
  run_name=${run_name}\
  save_folder=llama2-7b-20kcali-1.5x-finetune-30000batch_lr_1e-4_4096_fineweb\
  calibration_data_size=${calibration_data_size} \
  target=${target}\
  is_prune=${is_prune}\
  global_train_batch_size=${global_train_batch_size}\
  model.use_flash_attention_2=${use_flash_attn}