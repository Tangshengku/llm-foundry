CUDA_VISIBLE_DEVICES=2 python inference/convert_composer_to_hf.py \
  --composer_path /nfs/scistore19/alistgrp/stang/llm-foundry/srun_logs/evo_search_2.5x_reg_gradual_10B/ep0-ba8000-rank0.pt \
  --hf_output_path /nfs/scistore19/alistgrp/stang/llm-foundry/srun_logs/evo_search_2.5x_reg_gradual_10B/ep0-ba8000-rank0_hf\
  --output_precision bf16 \