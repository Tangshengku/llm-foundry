calibration_data_size=2048
target=2.5
is_prune=true
run_name=qwen2.5-3b_oneshot_pruning_1.25x
global_train_batch_size=1

# In some cases, bf16 may cause Non-Full rank during inverting the hessian matrix   
use_flash_attn=false

CUDA_VISIBLE_DEVICES=4,5,6,7 python train/train.py \
  train/yamls/pretrain/qwen2.5-3.yaml \
  run_name=${run_name}\
  save_folder=qwen2.5-3_timing\
  calibration_data_size=${calibration_data_size} \
  target=${target}\
  is_prune=${is_prune}\
  global_train_batch_size=${global_train_batch_size}\
  model.use_flash_attention_2=${use_flash_attn}