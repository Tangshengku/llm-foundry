# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming dataset conversion scripts for (hopefully) all datasets."""

from streaming.base.util import clean_stale_shared_memory
clean_stale_shared_memory()

import json
import os
import platform
from argparse import ArgumentParser, Namespace
from enum import Enum
from typing import Dict, Iterable, Optional, Union

import datasets as hf_datasets
import psutil
from streaming import MDSWriter
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from llmfoundry.data import ConcatTokensDataset, NoConcatDataset
from llmfoundry.utils.builders import build_tokenizer


class ConcatMode(Enum):
    NO_CONCAT = 'NO_CONCAT'
    CONCAT_TOKENS = 'CONCAT_TOKENS'


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Convert dataset into MDS format, optionally concatenating and tokenizing'
    )
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_subset',
                        type=str,
                        default=None,
                        help='E.g. "all" or "en"')
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'train_small', 'val', 'val_small', 'val_xsmall'])
    parser.add_argument('--out_root', type=str, required=True)
    parser.add_argument('--compression', type=str, default=None)

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '--concat_tokens',
        type=int,
        help='Convert text to tokens and concatenate up to this many tokens')

    parser.add_argument('--tokenizer', type=str, required=False, default=None)
    parser.add_argument('--tokenizer_kwargs', type=str, required=False)
    parser.add_argument('--bos_text', type=str, required=False, default=None)
    parser.add_argument('--eos_text', type=str, required=False, default=None)
    parser.add_argument('--no_wrap', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, required=False, default=None)
    parser.add_argument('--data_files',
                        type=str,
                        required=False,
                        default=None,
                        help='Useful to concat all data_subsets, e.g. with data_files="data/**/*"')
    parser.add_argument('--tokenizer_call_kwargs',
                        type=str,
                        required=False,
                        help="""These kwargs are passed to tokenizer directly in __call__.
                        This is useful when tokenizer_kwargs are completely ignored and it
                        is impossible to dig up the reason for it in the very cumbersome HF
                        codebase. For example, it is impossible to disable adding of special
                        tokens with Llama-3 tokenizer. It will always add BOS token, which we
                        sometimes do not want. Now we can disable in __call__ by setting add_special_tokens=False""")

    parsed = parser.parse_args()

    if parsed.tokenizer_kwargs is not None:
        parsed.tokenizer_kwargs = json.loads(parsed.tokenizer_kwargs)
    else:
        parsed.tokenizer_kwargs = {}

    if os.path.isdir(parsed.out_root) and len(
            set(os.listdir(parsed.out_root)).intersection(set(
                parsed.splits))) > 0:
        raise ValueError(
            f'--out_root={parsed.out_root} contains {os.listdir(parsed.out_root)} which cannot overlap with the requested splits {parsed.splits}.'
        )

    # Make sure we have needed concat options
    if (parsed.concat_tokens is not None and
            isinstance(parsed.concat_tokens, int) and parsed.tokenizer is None):
        parser.error(
            'When setting --concat_tokens, you must specify a --tokenizer')

    # now that we have validated them, change BOS/EOS to strings
    if parsed.bos_text is None:
        parsed.bos_text = ''
    if parsed.eos_text is None:
        parsed.eos_text = ''

    if parsed.tokenizer_call_kwargs is not None:
        parsed.tokenizer_call_kwargs = json.loads(parsed.tokenizer_call_kwargs)
    else:
        parsed.tokenizer_call_kwargs = {}

    return parsed


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
        data_files (str): useful to concat all data_subsets, e.g. with data_files="data/**/*"

    Returns:
        An IterableDataset.
    """
    hf_dataset = hf_datasets.load_dataset(path=dataset_name,
                                          name=data_subset,
                                          split=split,
                                          streaming=True,
                                          data_files=data_files)
    if mode == ConcatMode.NO_CONCAT:
        dataset = NoConcatDataset(hf_dataset)
    else:
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise ValueError(
                f'{tokenizer=} must be of type PreTrainedTokenizerBase')
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
        dataset = ConcatTokensDataset(hf_dataset=hf_dataset,
                                      tokenizer=tokenizer,
                                      max_length=max_length,
                                      bos_text=bos_text,
                                      eos_text=eos_text,
                                      no_wrap=no_wrap,
                                      hf_id=dataset_name,
                                      tokenizer_call_kwargs=tokenizer_call_kwargs)
    return dataset


def build_dataloader(dataset: Dataset, batch_size: int,
                     num_workers: Optional[int]) -> DataLoader:
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
    prefetch_factor = max(1, 2 * batch_size //
                          num_workers) if num_workers > 0 else 2

    return DataLoader(
        dataset=dataset,
        sampler=None,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )


def generate_samples(
        loader: DataLoader,
        truncate_num_samples: Optional[int] = None
) -> Iterable[Dict[str, bytes]]:
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
            yield {k: v[idx] for k, v in batch.items()}


def main(args: Namespace) -> None:
    if args.concat_tokens is not None:
        mode = ConcatMode.CONCAT_TOKENS
        tokenizer = build_tokenizer(args.tokenizer, args.tokenizer_kwargs)
        # we will enforce length, so suppress warnings about sequences too long for the model
        tokenizer.model_max_length = int(1e30)
        columns = {'tokens': 'bytes'}
    else:
        mode = ConcatMode.NO_CONCAT
        tokenizer = None
        columns = {'text': 'str'}

    out_root = args.out_root if args.data_subset is None else os.path.join(args.out_root, args.data_subset)
    dataset_info = ""
    for split_name in args.splits:
        hf_split = split_name   # TODO: unify these two
        folder_split = split_name

        dataset = build_hf_dataset(dataset_name=args.dataset,
                                   data_subset=args.data_subset,
                                   split=hf_split,
                                   mode=mode,
                                   max_length=args.concat_tokens,
                                   bos_text=args.bos_text,
                                   eos_text=args.eos_text,
                                   no_wrap=args.no_wrap,
                                   tokenizer=tokenizer,
                                   data_files=args.data_files,
                                   tokenizer_call_kwargs=args.tokenizer_call_kwargs)
        loader = build_dataloader(dataset=dataset,
                                  batch_size=512,
                                  num_workers=args.num_workers)
        samples = generate_samples(loader)

        print(f'Converting {folder_split} to MDS format...')
        print(
            f'Note: the progress bar is based on the dataset length before tokenization, and may finish at a value before 100%.'
        )

        n_samples = 0
        with MDSWriter(columns=columns,
                       out=os.path.join(out_root, folder_split),
                       compression=args.compression) as out:
            for sample in tqdm(samples, desc=folder_split):
                n_samples += 1
                out.write(sample)

        n_tokens_per_sample = args.concat_tokens if args.concat_tokens is not None else -1
        print(f'Total number of tokens in split={split_name} is: {n_samples} samples x {n_tokens_per_sample} tokens/sample = {n_samples * n_tokens_per_sample} tokens.')
        dataset_info += f'{split_name} = {n_samples * n_tokens_per_sample} tokens, seq_len = {args.concat_tokens}\n'

    print(f'Dataset info:\n{dataset_info}')
    with open(os.path.join(out_root, 'dataset_info.txt'), 'w') as f:
        f.write(dataset_info)


if __name__ == '__main__':
    main(parse_args())