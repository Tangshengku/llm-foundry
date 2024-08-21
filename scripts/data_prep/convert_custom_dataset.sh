#!/bin/bash

export ELDAR_DEBUG=1

python3 convert_custom_dataset.py \
  --dataset HuggingFaceFW/fineweb-edu --data_subset sample-10BT --splits train \
  --out_root /nfs/scistore19/alistgrp/stang/llm-foundry/tmp_data/finewebedu \
  --concat_tokens 4096 --tokenizer Shengkun/LLama2-7B-Structural-Prune-1.5x-32-20kCalib --no_wrap --compression zstd