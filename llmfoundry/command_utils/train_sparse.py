# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import gc
import logging
import os
import copy
import time
import warnings
from typing import Any, Optional, Union

import torch
import torch.distributed
from transformers import AutoModelForCausalLM
from composer import ComposerModel, Trainer
from composer.core.callback import Callback
from composer.profiler import (
    JSONTraceHandler,
    Profiler,
    TraceHandler,
    cyclic_schedule,
)
from composer.utils import dist, get_device, reproducibility
from omegaconf import DictConfig
from omegaconf import OmegaConf as om

from llmfoundry.callbacks import AsyncEval, HuggingFaceCheckpointer
from llmfoundry.data.dataloader import build_dataloader
from llmfoundry.eval.metrics.nlp import InContextLearningMetric
from llmfoundry.layers_registry import ffns_with_megablocks
from llmfoundry.utils import (
    find_mosaicml_logger,
    log_train_analytics,
    maybe_create_mosaicml_logger,
)
from llmfoundry.utils.builders import (
    add_metrics_to_eval_loaders,
    build_algorithm,
    build_callback,
    build_composer_model,
    build_evaluators,
    build_load_planner,
    build_logger,
    build_optimizer,
    build_save_planner,
    build_scheduler,
    build_tokenizer,
)
from llmfoundry.utils.config_utils import (
    TRAIN_CONFIG_KEYS,
    TrainConfig,
    log_config,
    log_dataset_uri,
    make_dataclass_and_log_config,
    pop_config,
    process_init_device,
)
from llmfoundry.utils.exceptions import (
    BaseContextualError,
    EvalDataLoaderLocation,
    TrainDataLoaderLocation,
)
from llmfoundry.utils.registry_utils import import_file
from llmfoundry.algorithms import KnowledgeDistillation

from .ziplm import *
from .timing import *

log = logging.getLogger(__name__)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def validate_config(train_config: TrainConfig):
    """Validates compatible model and dataloader selection."""
    # Validate the rest of the config
    loaders = [train_config.train_loader]
    if train_config.eval_loaders is not None:
        for loader in (train_config.eval_loaders or []):  # pyright
            if 'label' not in loader or loader['label'] is None:
                raise ValueError(
                    'When specifying multiple evaluation datasets, each one must include the \
                            `label` attribute.',
                )
            loaders.append(loader)
    if train_config.eval_loader is not None:
        loaders.append(train_config.eval_loader)
    for loader in loaders:
        if loader['name'] == 'text':
            if train_config.model['name'] == 'hf_t5':
                raise ValueError(
                    f'Model type "{train_config.model["name"]}" is not supported when using the "text " ' +\
                    f'dataloader. Only finetuning is supported.')

    if train_config.icl_tasks is not None or train_config.icl_tasks_str is not None:
        if train_config.model['name'] == 'hf_t5':
            raise ValueError(
                'ICL evaluation does not currently support Encoder-Decoder models, such as "hf_t5".',
            )

    if (
        train_config.model.get('fc_type', 'torch') != 'te' and
        'te' not in train_config.model.get('ffn_config',
                                           {}).get('ffn_type', 'mptmlp') and
        'fp8' in train_config.precision
    ):
        warnings.warn(
            "fp8 only supported for te.Linear layers. Either set `cfg.model.fc_typ='te'` or "
            +
            "`cfg.model.ffn_config.ffn_type='te_ln_mlp'` to enable layers using fp8 precision.",
        )

    if (
        train_config.model.get('fc_type', 'torch') == 'te' or 'te'
        in train_config.model.get('ffn_config', {}).get('ffn_type', 'mptmlp')
    ):
        fsdp_config = train_config.fsdp_config
        act_ckpt = fsdp_config.get(
            'activation_checkpointing',
            False,
        ) if fsdp_config else False
        act_ckpt_reentrant = fsdp_config.get(
            'activation_checkpointing_reentrant',
            False,
        ) if fsdp_config else False
        if fsdp_config is not None and act_ckpt == True and act_ckpt_reentrant == True:
            warnings.warn(
                '`te.Linear` layers do not support activation_checkpointing with '
                + '`activation_checkpointing_reentrant = True`. ' +
                'Setting cfg.fsdp_config.activation_checkpointing_reentrant=False.',
            )
            assert train_config.fsdp_config is not None  # pyright (this is known because fsdp_config is not None)
            train_config.fsdp_config['activation_checkpointing_reentrant'
                                    ] = False

    if train_config.model.get('ffn_config',
                              {}).get('ffn_type', 'mptmlp') == 'te_ln_mlp':
        warnings.warn(
            '`te.LayerNormMLP` requires has issues with torch._dynamo. ' +
            'Setting `torch._dynamo.config.suppress_errors = True` and falling back to eager.',
        )
        torch._dynamo.config.suppress_errors = True  # type: ignore (third-party)

    if train_config.model.get('load_in_8bit', False):
        raise ValueError(
            '`load_in_8bit` is only supported for evaluation rather than training.',
        )

    if train_config.model.get('ffn_config', {}).get(
        'ffn_type',
        'mptmlp',
    ) in ffns_with_megablocks:
        moe_world_size = train_config.model.get('ffn_config',
                                                {}).get('moe_world_size', 1)
        use_orig_params = train_config.fsdp_config.get(
            'use_orig_params',
            True,
        ) if train_config.fsdp_config is not None else True
        if moe_world_size > 1 and not use_orig_params:
            raise ValueError(
                f'MoEs with expert parallelism (moe_world_size {moe_world_size} > 1) require `use_orig_params=True`.',
            )


def _log_num_params(model: ComposerModel, logged_cfg: dict[str, Any]):
    # Log number of parameters
    if hasattr(model, 'n_total_params'):
        n_params = model.n_total_params
        n_trainable_params = n_params  # TODO: we currently assume all parameters are trainable.
    else:
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
    if hasattr(model, 'n_active_params'):
        n_active_params = model.n_active_params
    else:
        n_active_params = n_params
    logged_cfg.update({
        'n_params': n_params,
        'n_active_params': n_active_params,
        'n_trainable_params': n_trainable_params,
    })


def _initialize_dist_with_barrier(dist_timeout: Union[int, float]):
    """Initialize distributed and test setup with a barrier.

    Args:
        dist_timeout (Union[int, float]): Timeout for initializing the process group
    """
    log.debug('Initializing dist with device...')
    dist.initialize_dist(get_device(None), timeout=dist_timeout)
    log.debug('Testing barrier with device...')
    dist.barrier()
    log.debug('Barrier test passed with device.')


def train(cfg: DictConfig) -> Trainer:
    code_paths = cfg.get('code_paths', [])
    # Import any user provided code
    for code_path in code_paths:
        import_file(code_path)

    logged_cfg, train_cfg = make_dataclass_and_log_config(
        cfg,
        TrainConfig,
        TRAIN_CONFIG_KEYS,
        transforms='all',
    )

    # Set logging level
    if train_cfg.python_log_level is not None:
        logging.basicConfig(
            # Example of format string
            # 2022-06-29 11:22:26,152: rank0[822018][MainThread]: INFO: Message here
            format=
            f'%(asctime)s: rank{dist.get_global_rank()}[%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s',
        )
        logging.getLogger('llmfoundry').setLevel(
            train_cfg.python_log_level.upper(),
        )  # Foundry module
        logging.getLogger(__name__).setLevel(
            train_cfg.python_log_level.upper(),
        )  # Train script

    _initialize_dist_with_barrier(dist_timeout=train_cfg.dist_timeout)

    # Filter deprecation warning from torch internal usage
    warnings.filterwarnings(
        action='ignore',
        category=UserWarning,
        message=
        'torch.distributed.*_base is a private function and will be deprecated.*',
    )

    # Check for incompatibilities between the model and data loaders
    validate_config(train_cfg)

    cuda_alloc_conf = []
    # Get max split size mb
    max_split_size_mb: Optional[int] = train_cfg.max_split_size_mb
    if max_split_size_mb is not None:
        cuda_alloc_conf.append(f'max_split_size_mb:{max_split_size_mb}')

    # Expandable segments
    if train_cfg.expandable_segments:
        cuda_alloc_conf.append('expandable_segments:True')

    if len(cuda_alloc_conf) > 0:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ','.join(cuda_alloc_conf)

    # Set CUDA lazy loading
    # This can save a bit of memory if not all modules are needed
    cuda_load_lazy: bool = train_cfg.cuda_load_lazy
    if cuda_load_lazy:
        os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

    # Set seed first
    seed: int = train_cfg.seed
    reproducibility.seed_all(seed)

    # Mandatory model training configs
    model_config = train_cfg.model
    train_loader_config = train_cfg.train_loader

    # Optional fsdp data, fine-tuning, and eval configs
    fsdp_config: Optional[dict[str, Any]] = train_cfg.fsdp_config

    if fsdp_config is not None:
        if 'load_planner' in fsdp_config:
            load_planners = list(fsdp_config['load_planner'].items())
            if len(load_planners) > 1:
                raise ValueError(
                    'Only one load planner can be specified in the config.',
                )
            load_planner_name, load_planner_config = load_planners[0]
            fsdp_config['load_planner'] = build_load_planner(
                load_planner_name,
                **load_planner_config,
            )

        if 'save_planner' in fsdp_config:
            save_planners = list(fsdp_config['save_planner'].items())
            if len(save_planners) > 1:
                raise ValueError(
                    'Only one save planner can be specified in the config.',
                )
            save_planner_name, save_planner_config = save_planners[0]
            fsdp_config['save_planner'] = build_save_planner(
                save_planner_name,
                **save_planner_config,
            )

    eval_loader_config = train_cfg.eval_loader if train_cfg.eval_loader is not None else train_cfg.eval_loaders
    icl_tasks_config = train_cfg.icl_tasks or train_cfg.icl_tasks_str
    eval_gauntlet_config = train_cfg.eval_gauntlet or train_cfg.eval_gauntlet_str

    # Optional parameters will be set to default values if not specified.
    default_run_name: str = os.environ.get('RUN_NAME', 'llm')
    run_name: str = train_cfg.run_name if train_cfg.run_name else default_run_name
    is_state_dict_sharded: bool = (
        fsdp_config.get('state_dict_type', 'full') == 'sharded'
    ) if fsdp_config else False
    save_latest_filename: str = train_cfg.save_latest_filename if train_cfg.save_latest_filename else 'latest-sharded-rank{rank}' if is_state_dict_sharded else 'latest-rank{rank}.pt'
    save_filename: str = train_cfg.save_filename if train_cfg.save_filename else 'ep{epoch}-ba{batch}-rank{rank}.pt'

    # Enable autoresume from model checkpoints if possible
    autoresume_default: bool = False
    if logged_cfg.get('run_name', None) is not None \
        and train_cfg.save_folder is not None \
        and not train_cfg.save_overwrite \
        and not train_cfg.save_weights_only:
        autoresume_default = True

    if not train_cfg.autoresume and autoresume_default:
        log.info(
            'As run_name, save_folder, and save_latest_filename are set, \
                changing autoresume default to True...',
        )

    # Warn if fsdp is enabled but user only has 1 GPU
    if dist.get_world_size() == 1 and fsdp_config is not None:
        warnings.warn(
            'FSDP is not applicable for single-GPU training. Reverting to DDP.',
        )
        fsdp_config = None

    # Initialize context
    init_context = process_init_device(model_config, fsdp_config)
    logged_cfg.update({'fsdp_config': fsdp_config}, merge=True)

    from streaming.base.util import clean_stale_shared_memory
    clean_stale_shared_memory()
    # Build tokenizer
    log.info('Building tokenizer...')
    tokenizer_name = train_cfg.tokenizer['name']
    tokenizer_kwargs = train_cfg.tokenizer.get('kwargs', {})
    tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)

    # Scheduler
    scheduler_name: str = train_cfg.scheduler.pop('name')
    scheduler = build_scheduler(scheduler_name, train_cfg.scheduler)

    # Loggers
    loggers = [
        build_logger(str(name), logger_cfg)
        for name, logger_cfg in train_cfg.loggers.items()
    ] if train_cfg.loggers else []

    mosaicml_logger = find_mosaicml_logger(loggers)
    if mosaicml_logger is None:
        mosaicml_logger = maybe_create_mosaicml_logger()
        if mosaicml_logger is not None:
            # mosaicml_logger will be None if run isn't on MosaicML platform
            loggers.append(mosaicml_logger)

    if train_cfg.metadata is not None:
        # Optionally flatten the metadata for logging
        if train_cfg.flatten_metadata:
            logged_cfg.pop('metadata', None)
            common_keys = set(
                logged_cfg.keys(),
            ) & set(train_cfg.metadata.keys())
            if len(common_keys) > 0:
                raise ValueError(
                    f'Keys {common_keys} are already present in the config. Please rename them in metadata '
                    +
                    'or set flatten_metadata=False to avoid flattening the metadata in the logged config.',
                )

            logged_cfg.update(train_cfg.metadata, merge=True)

        if mosaicml_logger is not None:
            mosaicml_logger.log_metrics(train_cfg.metadata)
            mosaicml_logger._flush_metadata(force_flush=True)

    # Profiling
    profiler: Optional[Profiler] = None
    profiler_cfg = train_cfg.profiler
    if profiler_cfg:
        profiler_schedule_cfg: dict = pop_config(
            profiler_cfg,
            'schedule',
            must_exist=True,
        )
        profiler_schedule = cyclic_schedule(**profiler_schedule_cfg)
        # Only support json trace handler
        profiler_trace_handlers: list[TraceHandler] = []
        profiler_trace_cfg: Optional[dict] = pop_config(
            profiler_cfg,
            'json_trace_handler',
            must_exist=False,
            default_value=None,
        )
        if profiler_trace_cfg:
            profiler_trace_handlers.append(
                JSONTraceHandler(**profiler_trace_cfg),
            )
        profiler = Profiler(
            **profiler_cfg,
            trace_handlers=profiler_trace_handlers,
            schedule=profiler_schedule,
        )

    callback_configs = train_cfg.callbacks or {}

    # Callbacks
    callbacks: list[Callback] = [
        build_callback(
            name=str(name),
            kwargs=callback_cfg,
            train_config=logged_cfg,
        ) for name, callback_cfg in callback_configs.items()
    ]

    use_async_eval = any(isinstance(c, AsyncEval) for c in callbacks)

    algorithm_configs = train_cfg.algorithms or {}

    # Algorithms
    algorithms = [
        build_algorithm(str(name), algorithm_cfg)
        for name, algorithm_cfg in algorithm_configs.items()
    ]

    # Dataloaders
    log.info('Building train loader...')
    try:
        train_loader = build_dataloader(
            train_loader_config,
            tokenizer,
            train_cfg.device_train_batch_size,
        )
    except BaseContextualError as e:
        e.location = TrainDataLoaderLocation
        raise e

    if mosaicml_logger is not None:
        mosaicml_logger.log_metrics({'data_validated': time.time()})

    ## Evaluation
    if use_async_eval:
        evaluators = []
        if train_cfg.eval_first:
            warnings.warn(
                'AsyncEval callback does not support eval_first=True. Ignoring.',
            )
            train_cfg.eval_first = False

    else:
        try:
            log.info('Building eval loader...')
            eval_icl_seq_len: int = train_cfg.icl_seq_len if train_cfg.icl_seq_len else train_cfg.max_seq_len
            evaluators, _, eval_gauntlet_callback = build_evaluators(
                eval_loader_config,
                icl_tasks_config,
                eval_gauntlet_config,
                tokenizer=tokenizer,
                device_eval_batch_size=train_cfg.device_eval_batch_size,
                icl_seq_len=eval_icl_seq_len,
                icl_subset_num_batches=train_cfg.icl_subset_num_batches,
            )
            if eval_gauntlet_callback is not None:
                callbacks.append(eval_gauntlet_callback)
        except BaseContextualError as e:
            e.location = EvalDataLoaderLocation
            raise e

    if mosaicml_logger is not None:
        log_train_analytics(
            mosaicml_logger,
            model_config,
            train_loader_config,
            eval_loader_config,
            train_cfg.callbacks,
            tokenizer_name,
            train_cfg.load_path,
            icl_tasks_config,
            eval_gauntlet_config,
        )
    # Build Model

    from_pretrained_old = AutoModelForCausalLM.from_pretrained
    # Define new init
    def from_pretrained_overriden(*args, **kwargs):
        model = from_pretrained_old(*args, **kwargs)
        # load sparse checkpoint
        # model.load_state_dict(sparse_model.model.state_dict())
        state = torch.load("/nfs/scistore19/alistgrp/stang/llm-foundry/weights/evo_search_reg_2.0x/model_2x_mannul.pt", map_location="cpu")
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state, prefix='model.')
        model.load_state_dict(state)
        shrink(model=model, is_transformers=True)
        print("Model parameters after shrinking: {}".format(get_parameter_number(model)))

        state = torch.load("/nfs/scistore19/alistgrp/stang/llm-foundry/weights/evo_search_reg_2.5x_gradual/model_2.5x_mannul.pt", map_location="cpu")
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state, prefix='model.')
        model.load_state_dict(state)
        shrink(model=model, is_transformers=True)
        print("Model parameters after shrinking: {}".format(get_parameter_number(model)))
        return model

    # Override init
    AutoModelForCausalLM.from_pretrained = staticmethod(from_pretrained_overriden)

    log.info('Initializing model...')
    name = model_config.pop('name')
    assert isinstance(name, str)
    assert isinstance(model_config, dict)
    model = build_composer_model(
        name=name,
        tokenizer=tokenizer,
        init_context=init_context,
        master_weights_dtype=model_config.pop('master_weights_dtype', None),
        cfg=model_config,
    )

    if "knowledge_distillation" in cfg and cfg.knowledge_distillation.teacher_name_or_path is not None:
        print(f"[Debug: Knowledge Distillation] config = {cfg.knowledge_distillation}")
        teacher_config = copy.deepcopy(model_config)
        # print(teacher_config)
        teacher_config['pretrained_model_name_or_path'] = cfg.knowledge_distillation.teacher_name_or_path
        teacher = build_composer_model(
            name=name, cfg=teacher_config, tokenizer=tokenizer,
            init_context=init_context,
            master_weights_dtype=teacher_config.get('master_weights_dtype', None),)
        teacher.eval()
    
    print("Model parameters before shrinking: {}".format(get_parameter_number(model)))
    shrink(model=model)
    print("Model parameters after shrinking: {}".format(get_parameter_number(model)))
    # load_pruned_model(module=model, 
    #                   db_file="/nfs/scistore19/alistgrp/stang/llm-foundry/scripts/database_prune_with_fine_edu_20kcali_2x.db",
    #                   profile="/nfs/scistore19/alistgrp/stang/llm-foundry/scripts/profile_2.0_prune_with_fine_edu_20kcali_2x_4_layers.txt")
    # torch.save(model.state_dict(), \
    #                "/nfs/scistore19/alistgrp/stang/llm-foundry/weights/2k_fineweb_oneshot_4layers/model.pt")
    # return
    # Push the Finetuned Weight to Huggingface
    # state = torch.load("/nfs/scistore19/alistgrp/stang/llm-foundry/scripts/llama2-7b-20kcali-1.5x-finetune-30000batch_lr_1e-4_4096_fineweb/ep1-ba30000-rank0.pt")
    # model.load_state_dict(state["state"]["model"])
    # print("Finetuned model is loaded, Ready to prune")
    # model.tokenizer.push_to_hub("Shengkun/LLama2-7B-Structural-Prune-1.5x-32-20kCalib-Fineweb-Finetuned")
    # model.model.push_to_hub("Shengkun/LLama2-7B-Structural-Prune-1.5x-32-20kCalib-Fineweb-Finetuned")

    # state = torch.load("/nfs/scistore19/alistgrp/stang/llm-foundry/weights/evo_task_search_2.5/model.pt", map_location="cpu")
    # torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
    #     state, prefix='model.')
    # model.model.load_state_dict(state)
    # shrink(model=model)
    # print("Model parameters after shrinking: {}".format(get_parameter_number(model)))
    # # Some pushing codes, please just ignore
    # model.tokenizer.push_to_hub("Shengkun/evo_search_2x")
    # model.model.push_to_hub("Shengkun/evo_search_2x")
    # model.tokenizer.push_to_hub("Shengkun/Llama3-8B-Structural-Pruning-1.5")
    # model.model.push_to_hub("Shengkun/Llama3-8B-Structural-Pruning-1.5")

    timing = False
    if timing:
        timing_main(model, "cuda", train_loader.dataloader, is_bert=False)
        return

    if train_cfg.is_prune:
        oneshot_prune(train_loader.dataloader, model, target=train_cfg.target, loader_batchsize=1, \
                      loader_nsamples=train_cfg.calibration_data_size, timings_file=train_cfg.timing_file, \
                        run_name=run_name)
        # For gradual pruning, the weight might not be pushed to huggingface directly
        # So, a compromised solution is to save the pruned weights locally
        # Please uncomment this if you want to save the weight locally
        torch.save(model.state_dict(), \
                   "/nfs/scistore19/alistgrp/stang/llm-foundry/weights/evo_search/model.pt")
        return
    
    # Load the weight of pruned model
    # state = torch.load("/nfs/scistore19/alistgrp/stang/llm-foundry/weights/20kcali_gradual_from_1.5/model.pt")
    # torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
    #     state, prefix='model.')
    # model.model.load_state_dict(state)
    # shrink(model=model)
    # print("Model parameters after shrinking: {}".format(get_parameter_number(model)))
    # state = torch.load("/nfs/scistore19/alistgrp/stang/llm-foundry/weights/evo_search_within_group_2x/model.pt", map_location="cpu")
    # torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
    #     state, prefix='model.')
    # model.model.load_state_dict(state)
    # shrink(model=model)
    # print("Model parameters after shrinking: {}".format(get_parameter_number(model)))

    _log_num_params(model, logged_cfg)

    # Optimizer
    optimizer_name: str = train_cfg.optimizer.pop('name')
    optimizer_cfg = train_cfg.optimizer
    optimizer = build_optimizer(model, optimizer_name, optimizer_cfg)

    # Now add the eval metrics
    try:
        if eval_loader_config is not None and not use_async_eval:
            eval_metrics = model.get_metrics(is_train=False)
            non_icl_metrics = [
                metric_name for metric_name, metric in eval_metrics.items()
                if not isinstance(metric, InContextLearningMetric)
            ]
            evaluators = add_metrics_to_eval_loaders(
                evaluators,
                non_icl_metrics,
            )
    except BaseContextualError as e:
        e.location = EvalDataLoaderLocation
        raise e

    if "knowledge_distillation" in cfg and cfg.knowledge_distillation.teacher_name_or_path is not None:
        # assume algorithm is not None because we will always have `mask_pruned_weights`
        algorithms.append(KnowledgeDistillation(
            teacher,
            cfg.knowledge_distillation.temperature,
            cfg.knowledge_distillation.hardness_ce,
            cfg.knowledge_distillation.hardness_kldiv,
            cfg.knowledge_distillation.hardness_squarehead
            )
        )
    compile_config = train_cfg.compile_config
    # Build the Trainer
    log.info('Building trainer...')
    trainer = Trainer(
        run_name=run_name,
        seed=seed,
        model=model,
        train_dataloader=train_loader,
        train_subset_num_batches=train_cfg.train_subset_num_batches,
        eval_dataloader=evaluators,
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=train_cfg.max_duration,
        eval_interval=train_cfg.eval_interval,
        eval_subset_num_batches=train_cfg.eval_subset_num_batches,
        progress_bar=train_cfg.progress_bar,
        log_to_console=train_cfg.log_to_console,
        console_log_interval=train_cfg.console_log_interval,
        loggers=loggers,
        callbacks=callbacks,
        precision=train_cfg.precision,
        algorithms=algorithms,
        device_train_microbatch_size=train_cfg.device_train_microbatch_size,
        parallelism_config={'fsdp': fsdp_config},
        save_folder=train_cfg.save_folder,
        save_filename=save_filename,
        save_latest_filename=save_latest_filename,
        save_interval=train_cfg.save_interval,
        save_num_checkpoints_to_keep=train_cfg.save_num_checkpoints_to_keep,
        save_overwrite=train_cfg.save_overwrite,
        save_weights_only=train_cfg.save_weights_only,
        load_path=train_cfg.load_path,
        load_weights_only=train_cfg.load_weights_only,
        load_strict_model_weights=train_cfg.load_strict_model_weights,
        load_ignore_keys=train_cfg.load_ignore_keys,
        save_ignore_keys=train_cfg.save_ignore_keys,
        autoresume=train_cfg.autoresume,
        python_log_level=train_cfg.python_log_level,
        dist_timeout=train_cfg.dist_timeout,
        profiler=profiler,
        compile_config=compile_config,
        spin_dataloaders=train_cfg.spin_dataloaders,
    )

    # Optionally just save an HF checkpoint
    if train_cfg.only_hf_checkpoint:
        hf_checkpointer_callbacks = [
            c for c in callbacks if isinstance(c, HuggingFaceCheckpointer)
        ]
        if len(hf_checkpointer_callbacks) == 0:
            raise ValueError(
                'No HuggingFaceCheckpointer callback found, but only_hf_checkpoint was set to True. Please add a HuggingFaceCheckpointer.',
            )
        if len(hf_checkpointer_callbacks) > 1:
            raise ValueError(
                'Multiple HuggingFaceCheckpointer callbacks found, but only_hf_checkpoint was set to True. Please remove all but one HuggingFaceCheckpointer.',
            )

        hf_checkpointer_callback = hf_checkpointer_callbacks[0]
        hf_checkpointer_callback._save_checkpoint(trainer.state, trainer.logger)
        return trainer

    if train_cfg.only_composer_checkpoint:
        log.info('Not training. Only saving composer checkpoint.')
        trainer.save_checkpoint_to_save_folder()
        log.info('Done saving checkpoint.')
        return trainer

    if train_cfg.log_config:
        log.info('Logging config')
        log_config(trainer.logger, logged_cfg)
    log_dataset_uri(logged_cfg)
    torch.cuda.empty_cache()
    gc.collect()

    # Eval first if requested
    if train_cfg.eval_first and trainer.state.timestamp.batch.value == 0:
        trainer.eval()

    log.info('Starting training...')
    trainer.fit()

    log.info('Done.')
    return trainer


def train_from_yaml_sparse(
    yaml_path: str,
    args_list: Optional[list[str]] = None,
) -> Trainer:
    """Run the training with optional overrides from CLI."""
    # Load yaml and CLI arguments.
    om.clear_resolver('oc.env')
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    if args_list:
        cli_cfg = om.from_cli(args_list)
        yaml_cfg = om.merge(yaml_cfg, cli_cfg)
    assert isinstance(yaml_cfg, DictConfig)
    return train(yaml_cfg)