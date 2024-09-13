#!/bin/bash

export ELDAR_DEBUG=1

python3 convert_custom_pretraining_dataset.py \
  --dataset cais/mmlu --data_subset auxiliary_train --splits train \
  --out_root /nfs/scistore19/alistgrp/stang/llm-foundry/tmp_data/aux_mmlu \
  --concat_tokens 4096 --tokenizer meta-llama/Llama-2-7b-hf --no_wrap --compression zstd