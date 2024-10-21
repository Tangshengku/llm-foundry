CUDA_VISIBLE_DEVICES=0 python inference/convert_composer_to_hf.py \
  --composer_path /nfs/scistore19/alistgrp/stang/llm-foundry/srun_logs/evo_search_llama3.1_8b_2x_finetune_10B_1902_1e-4_randomseed_18/ep0-ba240-rank0.pt \
  --hf_output_path /nfs/scistore19/alistgrp/stang/llm-foundry/srun_logs/evo_search_llama3.1_8b_2x_finetune_10B_1902_1e-4_randomseed_18/ep0-ba240-rank0_hf\
  --output_precision bf16 \