# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming dataset conversion scripts for C4 and The Pile."""
from argparse import ArgumentParser, Namespace

from llmfoundry.command_utils import convert_custom_pretraining_dataset_from_args


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Convert dataset into MDS format, optionally concatenating and tokenizing',
    )
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument(
        '--data_subset',
        type=str,
        default=None,
        help='E.g. "all" or "en"',
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'train_small', 'val', 'val_small', 'val_xsmall'],
    )
    parser.add_argument('--out_root', type=str, required=True)
    parser.add_argument('--compression', type=str, default=None)

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '--concat_tokens',
        type=int,
        help='Convert text to tokens and concatenate up to this many tokens',
    )

    parser.add_argument('--tokenizer', type=str, required=False, default=None)
    parser.add_argument('--tokenizer_kwargs', type=str, required=False)
    parser.add_argument('--bos_text', type=str, required=False, default=None)
    parser.add_argument('--eos_text', type=str, required=False, default=None)
    parser.add_argument('--no_wrap', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, required=False, default=None)

    # custom args to handle special cases not supported by llm-foundry
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
    parser.add_argument('--add_position_ids', default=False, action='store_true', help="Useful to add position ids to each concatenated sample. The goal is to use this for variable length FlashAttention to avoid cross-document attention during pretraining")

    parsed = parser.parse_args()
    return parsed


if __name__ == '__main__':
    import os
    os.environ['CURL_CA_BUNDLE'] = ''

    args = parse_args()
    convert_custom_pretraining_dataset_from_args(
        dataset=args.dataset,
        data_subset=args.data_subset,
        splits=args.splits,
        out_root=args.out_root,
        compression=args.compression,
        concat_tokens=args.concat_tokens,
        tokenizer=args.tokenizer,
        tokenizer_kwargs=args.tokenizer_kwargs,
        bos_text=args.bos_text,
        eos_text=args.eos_text,
        no_wrap=args.no_wrap,
        num_workers=args.num_workers,
        data_files=args.data_files,
        tokenizer_call_kwargs=args.tokenizer_call_kwargs,
        add_position_ids=args.add_position_ids,
    )