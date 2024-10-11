#!/bin/bash

export ELDAR_DEBUG=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export FINEWEBEDU_THRESHOLD=0.9
# --data_subset sample-100BT 

python3 convert_custom_pretraining_dataset.py \
  --dataset /nfs/scistore19/alistgrp/stang/llm-foundry/tmp_data/finewebedu100B_raw/100BT   --splits train \
  --out_root /nfs/scistore19/alistgrp/stang/llm-foundry/tmp_data/finedwebedu_100B_llama3.1 \
  --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3.1-8B --no_wrap --compression zstd