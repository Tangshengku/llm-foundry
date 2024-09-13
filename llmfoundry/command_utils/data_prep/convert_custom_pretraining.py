# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming dataset conversion scripts for C4 and The Pile."""
import json
import os
import platform
from enum import Enum
from typing import Any, Dict, Iterable, Optional, Union

import datasets as hf_datasets
import psutil
import torch
from numpy.typing import NDArray
from streaming import MDSWriter
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from llmfoundry.data import ConcatTokensDataset, NoConcatDataset
from llmfoundry.utils.builders import build_tokenizer


class ConcatMode(Enum):
    NO_CONCAT = 'NO_CONCAT'
    CONCAT_TOKENS = 'CONCAT_TOKENS'


def build_hf_dataset(
    dataset_name: str,
    split: str,
    mode: ConcatMode,
    max_length: Optional[int] = None,
    bos_text: str = '',
    eos_text: str = '',
    no_wrap: bool = False,
    tokenizer: PreTrainedTokenizerBase = None,
    data_subset: Union[str, None] = None,
    data_files: Union[str, None] = None,
    tokenizer_call_kwargs: Optional[Dict] = None,
    add_position_ids: bool = False,
) -> IterableDataset:
    """Build an IterableDataset over the HF C4 or pile source data.

    Args:
        dataset_name (str): Dataset name
        split (str): Split name.
        mode (ConcatMode): NO_CONCAT, or CONCAT_TOKENS
        max_length (int): The length of concatenated tokens
        bos_text (str): text to insert at the beginning of each sequence
        eos_text (str): text to insert at the end of each sequence
        no_wrap (bool): if concatenating, whether to wrap text across `max_length` boundaries
        tokenizer (PreTrainedTokenizerBase): if mode is CONCAT_TOKENS, the tokenizer to use
        data_subset (str): Referred to as "name" in HuggingFace datasets.load_dataset.
            Typically "all" (The Pile) or "en" (c4).

    Returns:
        An IterableDataset.
    """
    hf_dataset = hf_datasets.load_dataset(
        path=dataset_name,
        name=data_subset,
        split=split,
        streaming=True,
        data_files=data_files,
    )
    if mode == ConcatMode.NO_CONCAT:
        dataset = NoConcatDataset(hf_dataset)
    else:
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise ValueError(
                f'{tokenizer=} must be of type PreTrainedTokenizerBase',
            )
        if max_length is None:
            raise ValueError(f'max_length must be set.')
        if bos_text + eos_text == '':
            test_tokens = tokenizer('test')
            if test_tokens['input_ids'][
                0] != tokenizer.bos_token_id and test_tokens['input_ids'][
                    -1] != tokenizer.eos_token_id:
                tok_error_msg = 'This tokenizer does not insert an EOS nor BOS token. '
                tok_error_msg += 'Concatenating with this tokenizer will result in sequences being '
                tok_error_msg += 'attached without a separating token. Please use another tokenizer, '
                tok_error_msg += 'such as facebook/opt-125m, or specify EOS/BOS text with e.g. '
                tok_error_msg += '--bos_text=<|endoftext|>.'
                raise ValueError(tok_error_msg)
        dataset = ConcatTokensDataset(
            hf_dataset=hf_dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            bos_text=bos_text,
            eos_text=eos_text,
            no_wrap=no_wrap,
            hf_id=dataset_name,
            tokenizer_call_kwargs=tokenizer_call_kwargs,
        )
    return dataset


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: Optional[int],
) -> DataLoader:
    if num_workers is None:
        # Multiple workers is only supported on linux machines
        if 'linux' or 'macos' in platform.platform().lower():
            num_workers = max(1, psutil.cpu_count())
        else:
            num_workers = 0

    # If using multiple workers, configure each worker to prefetch as many samples as it can, up to
    # the aggregate device batch size
    # If not using workers, the torch DataLoader expects the default value for prefetch_factor,
    # which non-intuitively must be 2.
    prefetch_factor = max(
        1,
        2 * batch_size // num_workers,
    ) if num_workers > 0 else 2

    return DataLoader(
        dataset=dataset,
        sampler=None,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )


def generate_samples(
    loader: DataLoader,
    truncate_num_samples: Optional[int] = None,
) -> Iterable[Union[dict[str, bytes], dict[str, NDArray]]]:
    """Generator over samples of a dataloader.

    Args:
       loader (DataLoader): A dataloader emitting batches like {key: [sample0_bytes, sample1_bytes, sample2_bytes, ...]}
       truncate_num_samples (Optional[int]): An optional # of samples to stop at.

    Yields:
        Sample dicts.
    """
    n_samples = 0
    for batch in loader:
        keys = list(batch.keys())
        current_bs = len(batch[keys[0]])
        for idx in range(current_bs):
            if truncate_num_samples is not None and n_samples == truncate_num_samples:
                return
            n_samples += 1
            yield {
                k:
                v[idx].numpy() if isinstance(v[idx], torch.Tensor) else v[idx]
                for k, v in batch.items()
            }


def convert_custom_pretraining_dataset(
    dataset: str,
    data_subset: Optional[str],
    splits: list[str],
    out_root: str,
    compression: Optional[str],
    concat_tokens: Optional[int],
    tokenizer: Optional[str],
    tokenizer_kwargs: dict[str, Any],
    bos_text: str,
    eos_text: str,
    no_wrap: bool,
    num_workers: Optional[int],
    data_files: Optional[str],
    tokenizer_call_kwargs: Optional[dict[str, Any]],
    add_position_ids: bool = False,
) -> None:
    """Converts HuggingFace datasets to MDS format.

    Args:
        dataset (str): Name of the dataset
        data_subset (Optional[str]): Subset of the dataset (e.g., "all" or "en")
        splits (list[str]): Comma-separated list of dataset splits
        out_root (str): Output root directory
        compression (Optional[str]): Compression type
        concat_tokens (Optional[int]): Concatenate tokens up to this many tokens
        tokenizer (Optional[str]): Tokenizer name
        tokenizer_kwargs (dict[str, Any]): Tokenizer keyword arguments
        bos_text (str): BOS text
        eos_text (str): EOS text
        no_wrap (bool): Do not wrap text across max_length boundaries
        num_workers (Optional[int]): Number of workers

    Raises:
        KeyError: If constants are not defined for the split
    """
    if concat_tokens is not None and tokenizer is not None:
        mode = ConcatMode.CONCAT_TOKENS
        built_tokenizer = build_tokenizer(tokenizer, tokenizer_kwargs)
        # we will enforce length, so suppress warnings about sequences too long for the model
        built_tokenizer.model_max_length = int(1e30)
        columns = {'tokens': 'ndarray:int32'}
        if add_position_ids:
            columns['position_ids'] = 'ndarray:int32'
    else:
        mode = ConcatMode.NO_CONCAT
        built_tokenizer = None
        columns = {'text': 'str'}

    out_root = out_root if data_subset is None else os.path.join(out_root, data_subset)
    dataset_info = ""
    for split_name in splits:
        hf_split = split_name
        folder_split = split_name

        # Get samples
        hf_dataset = build_hf_dataset(
            dataset_name=dataset,
            data_subset=data_subset,
            split=hf_split,
            mode=mode,
            max_length=concat_tokens,
            bos_text=bos_text,
            eos_text=eos_text,
            no_wrap=no_wrap,
            tokenizer=built_tokenizer,
            data_files=data_files,
            tokenizer_call_kwargs=tokenizer_call_kwargs,
            add_position_ids=add_position_ids,
        )
        loader = build_dataloader(
            dataset=hf_dataset,
            batch_size=512,
            num_workers=num_workers,
        )
        samples = generate_samples(loader)

        print(f'Converting {folder_split} to MDS format...')
        print(
            f'Note: the progress bar is based on the dataset length before tokenization, and may finish at a value before 100%.',
        )

        n_samples = 0
        with MDSWriter(
            columns=columns,
            out=os.path.join(out_root, folder_split),
            compression=compression,
        ) as out:
            for sample in tqdm(samples, desc=folder_split):
                n_samples += 1
                out.write(sample)

        n_tokens_per_sample = concat_tokens if concat_tokens is not None else -1
        print(f'Total number of tokens in split={split_name} is: {n_samples} samples x {n_tokens_per_sample} tokens/sample = {n_samples * n_tokens_per_sample} tokens.')
        dataset_info += f'{split_name} = {n_samples * n_tokens_per_sample} tokens, seq_len = {concat_tokens}\n'

    print(f'Dataset info:\n{dataset_info}')
    with open(os.path.join(out_root, 'dataset_info.txt'), 'w') as f:
        f.write(dataset_info)


def convert_custom_pretraining_dataset_from_args(
    dataset: str,
    data_subset: Optional[str],
    splits: list[str],
    out_root: str,
    compression: Optional[str],
    concat_tokens: Optional[int],
    tokenizer: Optional[str],
    tokenizer_kwargs: Optional[str],
    bos_text: Optional[str],
    eos_text: Optional[str],
    no_wrap: bool,
    num_workers: Optional[int],
    data_files: Optional[str],
    tokenizer_call_kwargs: Optional[str],
    add_position_ids: bool = False,
) -> None:
    """A wrapper for `convert_dataset_hf` that parses arguments.

    Args:
        dataset (str): Name of the dataset
        data_subset (Optional[str]): Subset of the dataset (e.g., "all" or "en")
        splits (list[str]): Comma-separated list of dataset splits
        out_root (str): Output root directory
        compression (Optional[str]): Compression type
        concat_tokens (Optional[int]): Concatenate tokens up to this many tokens
        tokenizer (Optional[str]): Tokenizer name
        tokenizer_kwargs (Optional[str]): Tokenizer keyword arguments in JSON format
        bos_text (Optional[str]): BOS text
        eos_text (Optional[str]): EOS text
        no_wrap (bool): Do not wrap text across max_length boundaries
        num_workers (Optional[int]): Number of workers

    Raises:
        ValueError: If the output directory already contains the requested splits
        ValueError: If `concat_tokens` is set but `tokenizer` is not
    """
    if tokenizer_kwargs:
        parsed_tokenizer_kwargs = json.loads(tokenizer_kwargs)
    else:
        parsed_tokenizer_kwargs = {}

    if tokenizer_call_kwargs:
        parsed_tokenizer_call_kwargs = json.loads(tokenizer_call_kwargs)
    else:
        parsed_tokenizer_call_kwargs = {}

    if os.path.isdir(out_root) and len(
        set(os.listdir(out_root)).intersection(set(splits)),
    ) > 0:
        raise ValueError(
            f'--out_root={out_root} contains {os.listdir(out_root)} which cannot overlap with the requested splits {splits}.',
        )

    # Make sure we have needed concat options
    if (
        concat_tokens is not None and isinstance(concat_tokens, int) and
        tokenizer is None
    ):
        raise ValueError(
            'When setting --concat_tokens, you must specify a --tokenizer',
        )

    # now that we have validated them, change BOS/EOS to strings and convert
    convert_custom_pretraining_dataset(
        dataset=dataset,
        data_subset=data_subset,
        splits=splits,
        out_root=out_root,
        compression=compression,
        concat_tokens=concat_tokens,
        tokenizer=tokenizer,
        tokenizer_kwargs=parsed_tokenizer_kwargs,
        bos_text=bos_text if bos_text else '',
        eos_text=eos_text if eos_text else '',
        no_wrap=no_wrap,
        num_workers=num_workers,
        data_files=data_files,
        tokenizer_call_kwargs=parsed_tokenizer_call_kwargs,
        add_position_ids=add_position_ids,
    )