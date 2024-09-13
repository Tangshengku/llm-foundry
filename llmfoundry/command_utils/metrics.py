import tqdm
import torch
import torch.nn.functional as F
import os
import logging

from transformers import AutoModelForCausalLM
from lm_eval import evaluator, utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.tasks import TaskManager


@torch.no_grad()
def compute_perplexity(model, data):
    # num_samples = len(data)
    device = next(model.parameters()).device
    # Running estimate of negative log-likelihood
    nll_running = 0.0
    # Number of tokens processed to far
    tokens_processed = 0
    # Loop through each batch
    for inputs in data:
        # j = min(i + batch_size, num_samples)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        # Forward pass through the model
        lm_logits = model(inputs).logits
        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs["labels"][:, 1:]
        # Compute loss
        loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        # Calculate negative log likelihood
        a = shift_labels.numel() / (tokens_processed + shift_labels.numel())
        b = tokens_processed / (tokens_processed + shift_labels.numel())
        nll_running = a * loss + b * nll_running
        # Update number of processed tokens
        tokens_processed += shift_labels.numel()
    # Compute perplexity
    ppl = nll_running.exp().item()
    return ppl

@torch.no_grad()
def compute_kl_div(model, data, target_logits):
    num_samples = len(data)
    device = next(model.parameters()).device
    # Running estimate of negative log-likelihood
    kl_div_running = 0
    # Number of tokens processed to far
    tokens_processed = 0
    # Loop through each batch
    for i, inputs in enumerate(data):
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        targets = target_logits[i].to(device)
        # Forward pass through the model
        lm_logits = model(inputs).logits
        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_targets = targets[:, :-1, :]
        # Compute loss
        loss = F.kl_div(
            shift_logits.reshape(-1, shift_logits.size(-1)).log_softmax(dim=-1),
            shift_targets.reshape(-1, shift_targets.size(-1)).log_softmax(dim=-1),
            log_target=True,
            reduction="batchmean",
        )
        # Calculate negative log likelihood
        a = shift_targets.numel() / (tokens_processed + shift_targets.numel())
        b = tokens_processed / (tokens_processed + shift_targets.numel())
        kl_div_running = a * loss + b * kl_div_running
        # Update number of processed tokens
        tokens_processed += shift_targets.numel()
    return kl_div_running.item()

@torch.no_grad()
def compute_mse(model, data, target_logits):
    num_samples = len(data)
    device = next(model.parameters()).device
    # Running estimate of negative log-likelihood
    mse_div_running = 0
    # Number of tokens processed to far
    tokens_processed = 0
    # Loop through each batch
    for i, inputs in enumerate(data):
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        targets = target_logits[i].to(device)
        # Forward pass through the model
        lm_logits = model(inputs).logits
        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_targets = targets[:, :-1, :]
        # Compute loss
        loss = F.mse_loss(
            shift_logits.reshape(-1, shift_logits.size(-1)).log_softmax(dim=-1),
            shift_targets.reshape(-1, shift_targets.size(-1)).log_softmax(dim=-1),
            reduction="mean",
        )
        # Calculate negative log likelihood
        a = shift_targets.numel() / (tokens_processed + shift_targets.numel())
        b = tokens_processed / (tokens_processed + shift_targets.numel())
        mse_div_running = a * loss + b * mse_div_running
        # Update number of processed tokens
        tokens_processed += shift_targets.numel()
    return mse_div_running.item()


@torch.no_grad()
def compute_task_metric(sparse_model, tasks="sciq", num_fewshot=0,
                             base_model="meta-llama/Llama-2-7b-hf", ):
    # Backup old init
    from_pretrained_old = AutoModelForCausalLM.from_pretrained
    eval_logger = utils.eval_logger
    verbosity = "INFO"
    eval_logger.setLevel(getattr(logging, f"{verbosity}"))
    eval_logger.info(f"Verbosity set to {verbosity}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    task_manager = TaskManager(verbosity, include_path=None)
    # sparse_model_checkpoint = args.sparse_model_checkpoint

    # Define new init
    def from_pretrained_overriden(*args, **kwargs):
        model = from_pretrained_old(*args, **kwargs)
        # load sparse checkpoint
        model.load_state_dict(sparse_model.model.state_dict())
        return model

    # Override init
    AutoModelForCausalLM.from_pretrained = staticmethod(from_pretrained_overriden)

    if tasks is None:
        task_names = ALL_TASKS
    elif tasks == "list":
        eval_logger.info("Available Tasks:\n - {}".format("\n - ".join(sorted(ALL_TASKS))))
        sys.exit()
    else:
        if os.path.isdir(tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            tasks_list = tasks.split(",")
            task_names = task_manager.match_tasks(tasks_list)
            for task in [task for task in tasks_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
            task_missing = [
                task for task in tasks_list if task not in task_names and "*" not in task
            ]  # we don't want errors if a wildcard ("*") task name was used

            if task_missing:
                missing = ", ".join(task_missing)
                eval_logger.error(
                    f"Tasks were not found: {missing}\n"
                    f"{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
                )
                raise ValueError(
                    f"Tasks not found: {missing}. Try `lm-eval --tasks list` for list of available tasks, or '--verbosity DEBUG' to troubleshoot task registration issues."
                )


    results = evaluator.simple_evaluate(
        model="hf",
        model_args="pretrained=meta-llama/Llama-2-7b-hf",
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=1,
        max_batch_size=1,
        device="cuda",
        use_cache=None,
        limit=0.2,
        task_manager=task_manager,
        check_integrity=False,
        write_out=False,
        log_samples=False,
        gen_kwargs=None,
    )
    acc = results["results"]["sciq"]["acc,none"]
    print("acc is : ", acc)
    return acc