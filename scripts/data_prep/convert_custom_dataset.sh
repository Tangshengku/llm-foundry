#!/bin/bash

export ELDAR_DEBUG=1

python3 convert_custom_dataset.py \
  --dataset HuggingFaceFW/fineweb-edu --data_subset sample-100BT --splits train \
  --out_root /nfs/scistore19/alistgrp/stang/llm-foundry/tmp_data/finewebedu_100B \
  --concat_tokens 4096 --tokenizer meta-llama/Llama-2-7b-hf --no_wrap --compression zstd