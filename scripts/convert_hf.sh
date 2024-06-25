CUDA_VISIBLE_DEVICES=2 python inference/convert_composer_to_hf.py \
  --composer_path /nfs/scistore19/alistgrp/stang/llm-foundry/scripts/llama2-7b-20kcali-2x-finetune-5000batch_lr_1e-5_2048_squarehead_dolly/ep2-ba5000-rank0.pt \
  --hf_output_path /nfs/scistore19/alistgrp/stang/llm-foundry/scripts/llama2-7b-20kcali-2x-finetune-5000batch_lr_1e-5_2048_squarehead_dolly/ep2-ba5000-rank0_hf \
  --output_precision bf16 \