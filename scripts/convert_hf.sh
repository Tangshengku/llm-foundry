CUDA_VISIBLE_DEVICES=2 python inference/convert_composer_to_hf.py \
  --composer_path /nfs/scistore19/alistgrp/stang/llm-foundry/srun_logs/shearedllama_pruned_2.7B_fineweb/ep5-ba20000-rank0.pt \
  --hf_output_path /nfs/scistore19/alistgrp/stang/llm-foundry/srun_logs/shearedllama_pruned_2.7B_fineweb/ep5-ba20000-rank0_hf_wrong\
  --output_precision bf16 \