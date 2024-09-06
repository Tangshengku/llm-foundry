CUDA_VISIBLE_DEVICES=2 python inference/convert_composer_to_hf.py \
  --composer_path /nfs/scistore19/alistgrp/stang/llm-foundry/scripts/evo_search_fineweb_with_KD/ep0-ba1600-rank0.pt \
  --hf_output_path /nfs/scistore19/alistgrp/stang/llm-foundry/scripts/evo_search_fineweb_with_KD/ep0-ba1600-rank0_hf\
  --output_precision bf16 \