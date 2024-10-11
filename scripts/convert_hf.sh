CUDA_VISIBLE_DEVICES=5 python inference/convert_composer_to_hf.py \
  --composer_path /nfs/scistore19/alistgrp/stang/llm-foundry/scripts/shearedllama_2.7B_fineweb_20B/ep0-ba1500-rank0.pt \
  --hf_output_path /nfs/scistore19/alistgrp/stang/llm-foundry/scripts/shearedllama_2.7B_fineweb_20B/ep0-ba1500-rank0_hf\
  --output_precision bf16 \