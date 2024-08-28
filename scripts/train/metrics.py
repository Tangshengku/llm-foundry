import tqdm
import torch
import torch.nn.functional as F


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
        # for k, v in inputs.items():
        #     inputs[k] = v.to(device)
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