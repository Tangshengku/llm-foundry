calibration_data_size=20480
max_seq_len=8192
target=2.5
is_prune=true
run_name=llama_3_1_8b_evo_search_20kcalib
global_train_batch_size=1

# In some cases, bf16 may cause Non-Full rank during inverting the hessian matrix   
use_flash_attn=false

CUDA_VISIBLE_DEVICES=1,2,3,4 python train/train.py \
  train/yamls/pretrain/llama3-8b.yaml \
  run_name=${run_name}\
  save_folder=llama3_1_8b_timing\
  calibration_data_size=${calibration_data_size} \
  max_seq_len=${max_seq_len}\
  target=${target}\
  is_prune=${is_prune}\
  global_train_batch_size=${global_train_batch_size}\
  model.use_flash_attention_2=${use_flash_attn}