import os
import copy
import numpy as np
import random
import torch
from torch import nn
from torch.nn import Module
from transformers.modeling_utils import prune_linear_layer
from typing import List, Optional, Tuple, Union

from .metrics import compute_perplexity, compute_kl_div, compute_task_metric, compute_mse



def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res


class NoAttention(Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        return (hidden_states, None, None)


class NoIntermediate(Module):
    def forward(self, hidden_states):
        return hidden_states


class NoOutput(Module):
    def forward(self, hidden_states):
        return hidden_states


def shrink(model, update_mask=False, is_transformers=False, kv_ignore=False):
    if is_transformers:
        layers = model.model.layers
    else:
        layers = model.model.model.layers
    for layer in layers:
        if not isinstance(layer.self_attn, NoAttention):
            weight = layer.self_attn.o_proj.weight
            if torch.all(weight == 0):
                layer.self_attn = NoAttention()
            else:
                mask = torch.all(
                    weight.t().reshape((-1, weight.shape[0] * layer.self_attn.head_dim)) == 0, 1
                )
                if update_mask:
                    mask_ = (~mask.unsqueeze(1)) * torch.ones_like(weight.t(), device=mask.device).reshape(
                            (-1, weight.shape[0] * layer.attn.head_size)) 
                    mask_ = mask_.reshape(-1, weight.shape[0])
                    # mask_ = torch.ones_like(mask_, device=mask_.device)
                    # layer.self_attn.o_proj.mask = mask_.t()
                    # layer.self_attn.in_proj_linear_q.mask = mask_
                    # layer.self_attn.in_proj_linear_k.mask = mask_
                    # layer.self_attn.in_proj_linear_v.mask = mask_

                    layer.self_attn.o_proj.register_buffer("mask", mask_.t(), persistent=False)
                    layer.self_attn.in_proj_linear_q.register_buffer("mask", mask_, persistent=False)
                    layer.self_attn.in_proj_linear_k.register_buffer("mask", mask_, persistent=False)
                    layer.self_attn.in_proj_linear_v.register_buffer("mask", mask_, persistent=False)
                else:
                    idx = []
                    count = 0
                    for i in range(mask.numel()):
                        pruned_heads = layer.self_attn.pruned_heads
                        while count in pruned_heads:
                            count += 1
                        if mask[i]:
                            idx.append(count)
                        count += 1
                    if torch.any(mask):
                        layer.self_attn.prune_heads(idx, kv_ignore=kv_ignore)
        if not isinstance(layer.mlp.down_proj, NoOutput):
            weight = layer.mlp.down_proj.weight
            if torch.all(weight == 0):
                layer.mlp.up_proj = NoIntermediate()
                layer.mlp.gate_proj = NoIntermediate()
                layer.mlp.down_proj = NoOutput()
            else:
                mask = torch.all(weight == 0, 0)
                if update_mask:
                    mask_ = (~mask.unsqueeze(0)) * torch.ones_like(weight, device=mask.device)
                    # layer.mlp.down_proj.mask = mask_
                    # layer.mlp.up_proj = mask_.t()
                    # layer.mlp.gate_proj = mask_.t()
                    
                    layer.mlp.down_proj.register_buffer("mask", mask_, persistent=False)
                    layer.mlp.up_proj.register_buffer("mask", mask_.t(), persistent=False)
                    layer.mlp.gate_proj.register_buffer("mask", mask_.t(), persistent=False)
                elif torch.any(mask):
                    idx = torch.nonzero(~mask).flatten()
                    layer.mlp.gate_proj = prune_linear_layer(layer.mlp.gate_proj, idx)
                    layer.mlp.up_proj = prune_linear_layer(layer.mlp.up_proj, idx)
                    layer.mlp.down_proj = prune_linear_layer(layer.mlp.down_proj, idx, dim=1)


class ZipLM:
    def __init__(self, layer, device):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        # Accumulate in double precision
        self.H = torch.zeros((self.columns, self.columns), device=device, dtype=torch.double)
        self.nsamples = 0

    def add_batch(self, inp, out):
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t().to(self.H.device)
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        self.H += 2 / self.nsamples * (inp.matmul(inp.t())).double()

    def invert(self, H, percentdamp=.01):
        try:
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        except RuntimeError:
            diagmean = torch.mean(torch.diag(H))
            print('Hessian not full rank.')
            tmp = (percentdamp * diagmean) * torch.eye(self.columns, device=H.device)
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H + tmp))
        return Hinv

    def prepare(self):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        H = self.H.float()
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        Hinv = self.invert(H)
        Losses = torch.zeros([self.rows, self.columns + 1], device=H.device)
        return W.to(H.device), H, Hinv.to(H.device), Losses

    def prune_struct(self, pruned, size=1):
        pruned = pruned[:]
        W, H, Hinv, Losses = self.prepare()

        count = self.columns // size
        Losses = torch.zeros(count + 1, device=H.device)
        mask = torch.zeros(count, device=H.device).bool()
        rangecount = torch.arange(count, device=H.device)
        rangecolumns = torch.arange(self.columns, device=H.device)

        res = []
        if 0 in pruned:
            res.append(self.layer.weight.data.clone())
            pruned = pruned[1:]
            print('   0 error 0.0')
            if not pruned:
                return res
        if size == 1:
            for dropped in range(count + 1):
                diag = torch.diagonal(Hinv)
                scores = torch.sum(W ** 2, 0) / diag
                scores[mask] = float('inf')
                j = torch.argmin(scores)
                Losses[dropped] = scores[j]
                row = Hinv[j, :]
                d = diag[j]
                W -= ((W[:, j] / d).unsqueeze(1)).matmul(row.unsqueeze(0))
                mask[j] = True
                W[:, mask] = 0
                while dropped + 1 == pruned[0]:
                    res.append(W.clone().reshape(self.layer.weight.shape))
                    print('%4d error' % pruned[0], torch.sum(Losses).item() / 2)
                    pruned.pop(0)
                    if not len(pruned):
                        break
                if not len(pruned):
                    break
                row /= torch.sqrt(d)
                Hinv -= row.unsqueeze(1).matmul(row.unsqueeze(0))
        else:
            mask1 = torch.zeros(self.columns, device=H.device).bool()
            for dropped in range(count + 1):
                blocks = Hinv.reshape(count, size, count, size)
                blocks = blocks[rangecount, :, rangecount, :]
                try:
                    invblocks = torch.cholesky_inverse(torch.linalg.cholesky(blocks))
                except:
                    invblocks = torch.linalg.pinv(blocks, hermitian=True)
                W1 = W.reshape((self.rows, count, size)).transpose(0, 1)
                lambd = torch.bmm(W1, invblocks)
                scores = torch.sum(lambd * W1, (1, 2))
                scores[mask] = float('inf')
                j = torch.argmin(scores)
                Losses[dropped] = scores[j]
                rows = Hinv[(size * j):(size * (j + 1)), :]
                d = invblocks[j]
                W -= lambd[j].matmul(rows)
                mask[j] = True
                mask1[(size * j):(size * (j + 1))] = True
                W[:, mask1] = 0
                while dropped + 1 == pruned[0]:
                    res.append(W.clone().reshape(self.layer.weight.shape))
                    print('%4d error' % pruned[0], torch.sum(Losses).item() / 2)
                    pruned.pop(0)
                    if not len(pruned):
                        break
                if not len(pruned):
                    break
                Hinv -= rows.t().matmul(d.matmul(rows))
                Hinv[rangecolumns[mask1], rangecolumns[mask1]] = 1

        return res

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()


def gen_transformerdb(
    filename,
    get_model, run, dataloader,
    dataloader_passes=1,
    sparsities=[], min_sparsity=0, max_sparsity=.99, delta_sparse=.1,
    headcount=12, headsize=64, fcdim=4*768,
    attname='attention.output.dense', fcname='output.dense', run_name=""
):
    modelp = get_model()
    modeld = get_model()
    modeld.to("cuda")
    modelp.to("cuda:1")
    layersp = find_layers(modelp)
    layersd = find_layers(modeld)

    if not sparsities:
        sparsities = []
        density = 1 - min_sparsity
        while density > 1 - max_sparsity:
            sparsities.append(1 - density)
            density *= 1 - delta_sparse

    ziplm = {}
    for i, name in enumerate(layersp):
        # if fcname not in name:
        #     continue
        # print(name)
        if fcname in name or attname in name:
            layer = layersp[name]
            if i < len(layersp)/2:
                ziplm[name] = ZipLM(layer, device="cuda:2")
            else:
                ziplm[name] = ZipLM(layer, device="cuda:3")

    def add_batch(name):
        def tmp(layer, inp, out):
            ziplm[name].add_batch(inp[0].data, out.data)
        return tmp
    handles = []
    for name in ziplm:
        handles.append(layersd[name].register_forward_hook(add_batch(name)))
    for i in range(dataloader_passes):
        for j, batch in enumerate(dataloader):
            print(i, j)
            with torch.no_grad():
                run(modeld, batch)
    for h in handles:
        h.remove()
    torch.cuda.empty_cache()

    def prundim(name, group_size=1):
        if attname in name:
            return headsize
        if fcname in name:
            return 1
        return 0

    db = {}
    for name in ziplm:
        print(name)
        size = prundim(name)
        if size > 0:
            print('Structured pruning ...')
            if attname in name:
                sparsities1 = [i / headcount for i in range(headcount)]
                print(f"shape is {ziplm[name].layer.weight.shape[1]}")
                remaining = ziplm[name].layer.weight.shape[1] // headsize
                sparsities1 = sparsities1[(headcount - remaining):]
                pruned = [i for i in range(remaining)]
            else:
                sparsities1 = sparsities
                remaining = ziplm[name].layer.weight.shape[1]
                pruned = [round((1 - s) * fcdim / 32) * 32 for s in sparsities1]
                pruned = [remaining - p for p in pruned if p <= remaining]
                sparsities1 = sparsities1[-len(pruned):]
            Ws = ziplm[name].prune_struct(pruned, size=size)
            db[name] = {('%.4f' % s): w.cpu().clone() for s, w in zip(sparsities1, Ws)}
            Ws = None
            torch.cuda.empty_cache()
        ziplm[name].free()

    # torch.save(db, filename)
    weight_path=f"/nfs/scistore19/alistgrp/stang/llm-foundry/scripts/database/{run_name}"
    if not os.path.exists(weight_path):
        os.mkdir(weight_path)
    for name in db:
        layer_dir = os.path.join(weight_path, name)
        if not os.path.exists(layer_dir):
            os.mkdir(layer_dir)
        for sparsity in list(db[name].keys()):
            torch.save(db[name][sparsity], f"{layer_dir}/{sparsity}.pt")
        torch.save(torch.zeros_like(db[name][sparsity]).cpu(), f"{layer_dir}/1.0000.pt")

class StructuredSPDY:
    def __init__(
        self,
        target,
        db, errors, baselinetime, prunabletime, timings,
        model, run, dataloader,
        dpbuckets=10000,
    ):
        self.target = target
        self.db = db
        self.run = run
        self.dpbuckets = dpbuckets

        self.modelp = model.to("cuda:0")
        self.layersp = find_layers(self.modelp)

        self.batches = []
        for batch in dataloader:
            self.batches.append(run(self.modelp, batch, retmoved=True))

        self.layers = list(db.layers())
        self.sparsities = [list(errors[self.layers[l]].keys()) for l in range(len(self.layers))]
        self.costs = [
            [errors[self.layers[l]][s] for s in self.sparsities[l]] for l in range(len(self.layers))
        ]
        self.timings = [
            [timings[self.layers[l]][s] for s in self.sparsities[l]] for l in range(len(self.layers))
        ]

        self.baselinetime = baselinetime
        self.prunabletime = prunabletime
        if self.baselinetime is None:
            self.baselinetime = self.prunabletime
        targettime = self.baselinetime / self.target - (self.baselinetime - self.prunabletime)
        best = sum(min(c) for c in self.timings)
        if self.prunabletime < self.baselinetime:
            print('Max target:', self.baselinetime / (best + self.baselinetime - self.prunabletime))
        self.bucketsize = targettime / self.dpbuckets

        for row in self.timings:
            for i in range(len(row)):
                row[i] = int(round(row[i] / self.bucketsize))
        print('Loss/Base:', self.get_loss(self.modelp))

    def dp(self, costs):
        DP = np.full((len(costs), self.dpbuckets + 1), float('inf'))
        PD = np.full((len(costs), self.dpbuckets + 1), -1)

        for sparsity in range(len(costs[0])):
            if costs[0][sparsity] < DP[0][self.timings[0][sparsity]]:
                DP[0][self.timings[0][sparsity]] = costs[0][sparsity]
                PD[0][self.timings[0][sparsity]] = sparsity
        for layer in range(1, len(DP)):
            for sparsity in range(len(costs[layer])):
                timing = self.timings[layer][sparsity]
                score = costs[layer][sparsity]
                if timing == 0:
                    tmp = DP[layer - 1] + score
                    better = tmp < DP[layer]
                    if np.sum(better):
                        DP[layer][better] = tmp[better]
                        PD[layer][better] = sparsity
                    continue
                if timing > self.dpbuckets:
                    continue
                tmp = DP[layer - 1][:-timing] + score
                better = tmp < DP[layer][timing:]
                if np.sum(better):
                    DP[layer][timing:][better] = tmp[better]
                    PD[layer][timing:][better] = sparsity

        score = np.min(DP[-1, :])
        timing = np.argmin(DP[-1, :])

        solution = []
        for layer in range(len(DP) - 1, -1, -1):
            solution.append(PD[layer][timing])
            timing -= self.timings[layer][solution[-1]]
        solution.reverse()
        return solution

    def gen_costs(self, coefs):
        return [
            [self.costs[i][j] * coefs[i] for j in range(len(self.costs[i]))] \
            for i in range(len(self.costs))
        ]

    def stitch_model(self, solution):
        model = copy.deepcopy(self.modelp.to("cpu")).to("cuda:1")
        layers = find_layers(model)
        config = {
            self.layers[i]: self.sparsities[i][solution[i]] for i in range(len(self.layers))
        }
        self.db.stitch(layers, config)
        shrink(model)
        return model

    @torch.no_grad()
    def get_loss(self, model):
        loss = 0
        for batch in self.batches:
            for k, v in batch.items():
                batch[k] = v.to(model.model.device)
            # For OpenOrca
            # batch = batch.to(model.model.device)
            loss += self.run(model, batch, loss=True)
        return loss / len(self.batches)

    def get_score(self, coefs):
        costs = self.gen_costs(coefs)
        solution = self.dp(costs)
        model = self.stitch_model(solution)
        return self.get_loss(model)

    def save_profile(self, coefs, filename=''):
        solution = self.dp(self.gen_costs(coefs))
        if filename:
            with open(filename, 'w') as f:
                for i in range(len(solution)):
                    f.write('%s %s\n' % (self.sparsities[i][solution[i]], self.layers[i]))
        else:
            for i in range(len(solution)):
                print('%s %s' % (self.sparsities[i][solution[i]], self.layers[i]))

    def score(self, filename):
        with open(filename, 'r') as f:
            solution = []
            i = 0
            for l in f.readlines():
                splits = l.split(' ')
                sparsity = splits[0]
                name = splits[1][:-1]
                while self.layers[i] != name:
                    solution.append(len(self.sparsities[i]) - 1)
                    i += 1
                j = self.sparsities[i].index(sparsity)
                solution.append(j)
                i += 1
        print('Speedup:', self.baselinetime / (
            self.baselinetime - self.prunabletime + \
            sum(t[s] for s, t in zip(solution, self.timings)) * self.bucketsize
        ))
        print('Loss/Pruned:', self.get_loss(self.stitch_model(solution)))

    def dpsolve(self, save=''):
        coefs = np.ones(len(self.layers))
        print('Loss/Pruned:', self.get_score(coefs))
        self.save_profile(coefs)
        if save:
            self.save_profile(coefs, save)

    def search(
        self, save='', randinits=100, searchsteps=500, muteprob=.1
    ):
        print('Random inits ...')
        candidates = []
        for i in range(randinits):
            coefs = np.zeros(len(self.layers))
            for j in range(len(coefs)):
                coefs[j] = random.random()
            score = self.get_score(coefs)
            candidates.append((score, coefs))
            print('%04d  %.4f %.4f' % (i, min(c[0] for c in candidates), score))
        candidates.sort(key=lambda c: c[0])

        print('Local search ...')
        score, coefs = candidates[0]
        for i in range(searchsteps):
            coefs1 = coefs.copy()
            for j in range(len(coefs)):
                if random.random() < muteprob:
                    coefs1[j] = random.random()
            score1 = self.get_score(coefs1)
            print('%04d  %.4f %.4f' % (i, score, score1))
            if score1 < score:
                score = score1
                coefs = coefs1

        self.save_profile(coefs)
        if save:
            self.save_profile(coefs, save)

class StructuredEvoSearch:
    def __init__(self, model, db, calibration_dataloader, weight_path=None, fitness_fn="ppl") -> None:
        self.db = db
        self.data = []
        self.weight_path=weight_path
        self.level2attn_sparsity = dict()
        self.level2mlp_sparsity = dict()
        self.attnlevel2sparsity = dict()
        self.mlplevel2sparsity = dict()
        self.target_logits = []
        device = next(model.parameters()).device
        for inputs in calibration_dataloader:
            self.data.append(inputs)
            if fitness_fn in ["kl", "mse"]:
                for k, v in inputs.items():
                    inputs[k] = v.to(device)
                self.target_logits.append(model(inputs).logits.cpu())

    def generate_offspring(self, parent, layer_names, offspring_num, within_group=False,
                            group_num=8, group_index=0,
                           max_level=10, max_total_deviation=9999):
        
        print(f"Parent: {parent}")
        group_size = int(len(parent) / group_num)

        offspring_list = []
        offspring_list.append(parent)  # Elitist EA

        while len(offspring_list) < offspring_num:
            print(f"Start to generate {len(offspring_list)} offspring.")
            offspring = copy.deepcopy(parent)
            # mutate offspring
            num_flips = min(random.randint(1, int(group_size/2)), random.randint(1, int(group_size/2)))  # bias towards lower values
            for _ in range(num_flips):
                
                if within_group:
                    # positions where sparsity of attn can be decreased
                    while True:
                        attn_decr_id = random.randint(0, len(offspring)/ (2 * group_num) - 1)
                        layer_name = layer_names[attn_decr_id*2 + group_index * group_size]
                        level = offspring[attn_decr_id*2 + group_index * group_size]
                        if level - 1 < 0:
                            continue
                        weight_ = os.path.join(self.weight_path, layer_name, f"{self.level2attn_sparsity[str(level - 1)]}.pt")
                        if os.path.exists(weight_):
                            break

                        # if self.db.level2attn_sparsity[str(level - 1)] in self.db.db[layer_name].keys():
                        #     break 
                    # positions where sparsity of mlp can be decreased
                    while True:
                        mlp_decr_id = random.randint(0, len(offspring)/(2 * group_num) - 1)
                        layer_name = layer_names[mlp_decr_id*2 + group_index * group_size + 1]
                        level = offspring[mlp_decr_id*2 + group_index * group_size + 1]
                        if level - 1 < 0:
                            continue
                        weight_ = os.path.join(self.weight_path, layer_name, f"{self.level2mlp_sparsity[str(level - 1)]}.pt")
                        if  os.path.exists(weight_):
                            break
                        # if self.db.level2mlp_sparsity[str(level - 1)] in self.db.db[layer_name].keys():
                        #     break 
                    # positions where sparsity of attn can be increased
                    while True:
                        attn_incr_id = random.randint(0, len(offspring)/(2 * group_num) - 1)
                        layer_name = layer_names[attn_incr_id*2 + group_index * group_size]
                        level = offspring[attn_incr_id*2 + group_index * group_size]
                        if level + 1 > max_level:
                            continue
                        weight_ = os.path.join(self.weight_path, layer_name, f"{self.level2attn_sparsity[str(level + 1)]}.pt")
                        if  os.path.exists(weight_):
                            break
                        # if self.db.level2attn_sparsity[str(level + 1)] in self.db.db[layer_name].keys():
                        #     break
                    # positions where sparsity of mlp can be increased
                    while True:
                        mlp_incr_id = random.randint(0, len(offspring)/(2 * group_num) - 1)
                        layer_name = layer_names[mlp_incr_id*2 + group_index * group_size + 1]
                        level = offspring[mlp_incr_id*2 + group_index * group_size + 1]
                        if level + 1 > max_level:
                            continue
                        weight_ = os.path.join(self.weight_path, layer_name, f"{self.level2mlp_sparsity[str(level + 1)]}.pt")
                        if  os.path.exists(weight_):
                            break
                        # if self.db.level2mlp_sparsity[str(level + 1)] in self.db.db[layer_name].keys():
                        #     break
                    offspring[attn_decr_id*2 + group_index * group_size] -= 1
                    offspring[attn_incr_id*2 + group_index * group_size] += 1
                    offspring[mlp_decr_id*2 + group_index * group_size + 1] -= 1
                    offspring[mlp_incr_id*2 + group_index * group_size + 1] += 1    
                else:
                    # Group to decrease
                    while True:
                        group_decr_id = random.randint(0, group_num - 1)
                        layer_names = layer_names[group_decr_id * group_size : (group_decr_id+1) * group_size]
                        levels = offspring[group_decr_id * group_size : (group_decr_id+1) * group_size]
                        if min(levels) - 1 < 0:
                            continue
                        for layer_name, level in zip(layer_names, levels):
                            weight_ = os.path.join(self.weight_path, layer_name, f"{self.level2attn_sparsity[str(level - 1)]}.pt")
                            if  not os.path.exists(weight_):
                                continue
                        break
                        # if self.db.level2attn_sparsity[str(level - 1)] in self.db.db[layer_name].keys():
                        #     break 
                    # Group to increase
                    while True:
                        group_incr_id = random.randint(0, group_num - 1)
                        layer_names = layer_names[group_incr_id * group_size : (group_incr_id+1) * group_size]
                        levels = offspring[ group_incr_id * group_size : (group_incr_id+1) * group_size]
                        if max(levels) + 1 > max_level:
                            continue
                        for layer_name, level in zip(layer_names, levels):
                            weight_ = os.path.join(self.weight_path, layer_name, f"{self.level2attn_sparsity[str(level + 1)]}.pt")
                            if  not os.path.exists(weight_):
                                continue
                        break
                        # if self.db.level2attn_sparsity[str(level + 1)] in self.db.db[layer_name].keys():
                        #     break
                    for i in range(group_size):
                        offspring[group_decr_id * group_size + i] -= 1
                        offspring[group_incr_id * group_size + i] += 1

            # avoid duplicates
            if offspring in offspring_list:
                print("Duplicate offsprings")
                continue
            # skip if total deviation exceeds specified threshold
            if sum(map(abs, offspring)) > max_total_deviation:
                print("Exceed max deviation")
                continue
            offspring_list.append(offspring)


        return offspring_list
    
    def compute_fitness(self, model, data, fitness_fn, target_logits: Optional[torch.Tensor] = None, 
                        memory_efficient: bool = False) -> float:
        if fitness_fn == "ppl":
            # if memory_efficient:
            #     return compute_perplexity_layer_per_layer(model, data)
            return compute_perplexity(model, data)
        elif fitness_fn == "task":
                 return compute_task_metric(model)
        elif fitness_fn == "mse":
            return compute_mse(model, data, target_logits)
        else:
            return compute_kl_div(model, data, target_logits)
            return NotImplementedError

    def load_weight(self, model, layer_names, parent, weight_path):
        assert hasattr(model, "state")
        for layer_name, new_level, old_level in zip(layer_names, parent, model.state):
            if new_level != old_level:
                layer = model.get_submodule(layer_name)
                if "attn" in layer_name:
                    sparsity = self.level2attn_sparsity[str(new_level)]
                else:
                    sparsity = self.level2mlp_sparsity[str(new_level)]
                layer.weight.data = torch.load(os.path.join(
                    weight_path, layer_name, f"{sparsity}.pt"), map_location=model.model.device).to(layer.weight.dtype)
        # Update model state
        model.state = parent

    def get_berttimings(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
            baselinetime = float(lines[1])
            prunabletime = float(lines[3])
            i = 5
            attention = {}
            while ' ' in lines[i]:
                time, sparsity, level = lines[i].strip().split(' ')
                attention[sparsity] = float(time)
                self.level2attn_sparsity[level] = sparsity
                self.attnlevel2sparsity[sparsity] = level
                i += 1
            fc = {}
            i += 1
            while i < len(lines):
                time, sparsity, level = lines[i].strip().split(' ')
                fc[sparsity] = float(time)
                self.level2mlp_sparsity[level] = sparsity
                self.mlplevel2sparsity[sparsity] = level
                i += 1
        timings = {}
        # for name in self.db:
        #     # if "down_proj" not in name:
        #     #     continue
        #     timings[name] = attention if ('attention' in name or 'attn' in name) else fc
        #     # timings[name] = fc
        return baselinetime, prunabletime, timings

    def selection(self, model, parent, offspring_list,
                  layer_names, survivors_per_selection=[4, 2, 1], samples_per_selection=[2, 4, 8], fitness_fn="ppl",
                    add_parent_to_last_selection=False):
        
        if fitness_fn == "task":
            fitnesses = []
            for i, candidate in enumerate(offspring_list):
                # self.db.load_level_layers(model, layer_names, candidate)
                self.load_weight(model, layer_names, candidate, self.weight_path)
                fitness = self.compute_fitness(model, None, fitness_fn, None)
                fitness += sum(candidate[:4]) / 10
                fitnesses.append(fitness)
                print(f"Candidate {i} is evaluated.")
            # Keep only best
            best_ids = np.argsort(fitnesses)[-1]
            offspring_list, train_fitnesses = [offspring_list[best_ids]], [fitnesses[best_ids]]
        else:
            for num_survive, num_sample in zip(survivors_per_selection, samples_per_selection):
                # If specified, add parent to last_selection if not present
                if add_parent_to_last_selection:
                    if parent not in offspring_list:
                        offspring_list.append(parent)

                target_logits_minibatch = None
                fitnesses = []

                data = self.data[:num_sample]
                target_logits_minibatch = self.target_logits[:num_sample]
                for i, candidate in enumerate(offspring_list):
                    # self.db.load_level_layers(model, layer_names, candidate)
                    self.load_weight(model, layer_names, candidate, self.weight_path)
                    
                    # ppl +  sparsity regularizer
                    fitness = self.compute_fitness(model, data, fitness_fn, target_logits_minibatch)
                    for j, level in enumerate(candidate):
                        idx = int(j/ 4) 
                        # 0.2 For llama2, 0.1 for llama3.1
                        fitness += candidate[j] * (torch.exp(torch.tensor(-5 * idx))).item() * 0.1 # TODO: Choose the best sparsity regularizer 
                        fitness += torch.exp(0.5 * torch.tensor(candidate[j])) * 0.01

                    fitnesses.append(fitness)
                    print(f"Candidate {i} is evaluated.")
                # Keep only best
                best_ids = np.argsort(fitnesses)[:num_survive]
                offspring_list, train_fitnesses = [offspring_list[i] for i in best_ids], [fitnesses[i] for i in best_ids]
                # In the end we have lists with single element
        train_fitness = train_fitnesses[0]
        parent = offspring_list[0]
        return parent, train_fitness

class StructDatabase:
    def __init__(self, path, dense):
        self.db = torch.load(path)
        self.level2attn_sparsity = dict()
        self.level2mlp_sparsity = dict()
        denselayers = find_layers(dense)
        dev = next(iter(denselayers.values())).weight.device
        # dev = "cpu"
        for name in self.db:
            for sparsity in list(self.db[name].keys()):
                self.db[name][sparsity] = self.db[name][sparsity].to(dev)
        for name in self.db:
            self.db[name]['1.0000'] = torch.zeros_like(denselayers[name].weight.data).to(dev)
        sd = dense.state_dict()
        # self.biases = {n: sd[n + '.bias'] for n in self.db}

    def layers(self):
        return list(self.db.keys())

    def load(self, layers, name, config='0.0000', sd=None):
        if sd is not None:
            layers[name].weight.data = sd[name + '.weight'].to(layers[name].weight.device)
            # layers[name].weight.data = sd[name + '.weight'].to(layers[name].weight.device).to(torch.bfloat16)
            # layers[name].bias.data = sd[name + '.bias']
            return
        if isinstance(layers, dict):
            layers[name].weight.data = self.db[name][config].to(layers[name].weight.device)
        else:
            layer = layers.get_submodule(name)
            layer.weight.data = self.db[name][config].to(layer.weight.dtype).to(layer.weight.device)
        # layers[name].weight.data = self.db[name][config].to(layers[name].weight.device).to(torch.bfloat16)
        # layers[name].bias.data = self.biases[name]
        # if config == '1.0000':
        #     layers[name].bias.data = torch.zeros_like(layers[name].bias.data)

    def load_level_layers(self, model, layer_names, level_list):
        for layer_name, level in zip(layer_names, level_list):
            if "attn" in layer_name:
                self.load(model, layer_name, self.level2attn_sparsity[str(level)])
            elif "mlp" in layer_name:
                self.load(model, layer_name, self.level2mlp_sparsity[str(level)])

    def stitch(self, layers, config):
        for name in config:
            self.load(layers, name, config[name])

    def load_file(self, model, profile):
        config = {}
        with open(profile, 'r') as f:
            for line in f.readlines():
                splits = line.split(' ')
                sparsity = splits[0]
                name = splits[1][:-1]
                config[name] = sparsity
        for name in self.db:
            if name not in config:
                config[name] = '1.0000'
        layers = find_layers(model)
        self.stitch(layers, config)

    def load_errors(self, path):
        errors = {}
        with open(path, 'r') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                name = lines[i].strip()
                errors[name] = {}
                i += 1
                for _ in range(len(self.db[name])):
                    err, level = lines[i].strip().split(' ')
                    errors[name][level] = float(err)
                    i += 1
        return errors

    def get_berttimings(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
            baselinetime = float(lines[1])
            prunabletime = float(lines[3])
            i = 5
            attention = {}
            while ' ' in lines[i]:
                time, sparsity, level = lines[i].strip().split(' ')
                attention[sparsity] = float(time)
                self.level2attn_sparsity[level] = sparsity
                i += 1
            fc = {}
            i += 1
            while i < len(lines):
                time, sparsity, level = lines[i].strip().split(' ')
                fc[sparsity] = float(time)
                self.level2mlp_sparsity[level] = sparsity
                i += 1
        timings = {}
        for name in self.db:
            # if "down_proj" not in name:
            #     continue
            timings[name] = attention if ('attention' in name or 'attn' in name) else fc
            # timings[name] = fc
        return baselinetime, prunabletime, timings


def compute_pnorm(p, db, get_model, dataloader, run, filename):
    modeld = get_model().to("cuda")
    modelp = get_model().to("cuda:1")
    layersd = find_layers(modeld)
    layersp = find_layers(modelp)

    errs = {n: {} for n in db.layers()}
    def accumerrs(name):
        def tmp(layer, inp, out):
            errs[name]['dense'] = errs[name].get('dense', 0) + torch.sum(torch.abs(out.data) ** p).item()
            for config in sorted(db.db[name]):
                db.load(layersp, name, config)
                errs[name][config] = errs[name].get(config, 0) + (torch.sum(torch.abs(layersp[name](inp[0].to("cuda:1").data) - out.to("cuda:1").data) ** p)).to("cuda").item()
        return tmp
    for name in db.layers():
        layersd[name].register_forward_hook(accumerrs(name))

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            print(i)
            run(modeld, batch)

    with open(filename, 'w') as f:
        for name in errs:
            f.write(name + '\n')
            for config in sorted(errs[name]):
                if config != 'dense':
                    f.write('%.6f %s\n' % (errs[name][config] / errs[name]['dense'], config))

def compute_squared(db, get_model, dataloader, run, filename):
    compute_pnorm(2, db, get_model, dataloader, run, filename)


# def _dataloader_builder(dataloader, batchsize=16, nsamples=1024):
#     default_loader = trainer.get_train_dataloader()
#     template = dict(default_loader.__dict__)

#     # drop attributes that will be auto-initialized
#     to_drop = [k for k in template if k.startswith("_") or k == "batch_sampler"]
#     for item in to_drop:
#         template.pop(item)

#     # shuffle dataset and select nsamples from it
#     shuffled_dataset = template['dataset'].shuffle(seed=42)
#     nsamples = len(shuffled_dataset) if nsamples == -1 else nsamples
#     shuffled_dataset = shuffled_dataset.select(range(nsamples))

#     kwargs = {
#         'batch_size': batchsize,
#         'dataset': shuffled_dataset,
#         'sampler': torch.utils.data.RandomSampler(shuffled_dataset)
#     }
#     template.update(kwargs)
#     data_loader = type(default_loader)(**template)

#     for sample in data_loader:
#         sample = trainer._prepare_inputs(sample)
#         yield sample
    
def _dataloader_builder(dataloader, batchsize=256, nsamples=2048):
    # default_loader = dataloader
    # template = dict(default_loader.__dict__)

    # # drop attributes that will be auto-initialized
    # to_drop = [k for k in template if k.startswith("_") or k == "batch_sampler"]
    # for item in to_drop:
    #     template.pop(item)

    # # shuffle dataset and select nsamples from it
    # # shuffled_dataset = template['dataset'].shuffle(seed=42)
    # # nsamples = len(shuffled_dataset) if nsamples == -1 else nsamples
    # # shuffled_dataset = shuffled_dataset.select(range(nsamples))
    # template["pipeline"][0].dataset.pipeline[-1] = wds.batched(batchsize, partial=False)
    # dataset_new = wds.DataPipeline(template["pipeline"][0].dataset.pipeline)
    # kwargs = {
    #     'batch_size': None,
    #     'dataset': dataset_new,
    # }
    # template.update(kwargs)
    # data_loader = type(default_loader)(**kwargs)

    # data_loader.num_batches = math.ceil(nsamples / batchsize)
    # data_loader.num_samples = nsamples

    for i, sample in  enumerate(dataloader):
        # sample = dataloader._prepare_inputs(sample)
        if (i + 1) * (batchsize) <= nsamples:
            yield sample
        else:
            break

@torch.no_grad()
def _get_model(module):
    def foo():
        res = copy.deepcopy(module)
        res.eval()
        return res
    return foo

@torch.no_grad()
def _run_llama(model, batch, loss=False, retmoved=False):
    dev = next(iter(model.parameters())).device
    for k, v in batch.items():
        batch[k] = v.to(dev)
    if retmoved:
        return batch
    out = model(batch)
    if loss:
        return out.loss.item()
    return out
    # return torch.cat([out[key] for key in ['start_logits', 'end_logits']])

@torch.no_grad()
def oneshot_prune(dataloader, module: Module, target: float, loader_batchsize: int, loader_nsamples: int, timings_file: str, run_name: str):
    db_file = f'database_{run_name}.db'
    # module.to("cuda:2")
    module.to(torch.bfloat16)
    # gen_transformerdb(
    #     db_file,
    #     _get_model(module),
    #     _run_llama,
    #     _dataloader_builder(
    #         dataloader,
    #         batchsize=loader_batchsize,
    #         nsamples=loader_nsamples,
    #     ),
    #     headcount=module.config.num_attention_heads,
    #     headsize=module.config.hidden_size // module.config.num_attention_heads,
    #     fcdim=module.config.intermediate_size if hasattr(module.config, 'intermediate_size') else module.config.hidden_size * 4,
    #     attname='self_attn.o_proj',
    #     fcname='mlp.down_proj',
    #     run_name=run_name)
    
    # return
    # model = _get_model(module)()
    # db = StructDatabase(db_file, model)
    # error_file = f'errors_squared_{run_name}.txt'
    # compute_squared(
    #     db,
    #     _get_model(module),
    #     _dataloader_builder(
    #         dataloader,
    #         batchsize=loader_batchsize,
    #         nsamples=1024, # Please adjust this number for efficiency
    #     ),
    #     _run_llama,
    #     error_file
    # )
    # torch.cuda.empty_cache()

    # errors = db.load_errors(error_file)
    # baselinetime, prunabletime, timings = db.get_berttimings(timings_file)
    # struct_spdy = StructuredSPDY(
    #     target, db, errors, baselinetime, prunabletime, timings,
    #     module, _run_llama,
    #     _dataloader_builder(
    #         dataloader,
    #         batchsize=1,
    #         nsamples=20,
    #     ),
    # )

    # profile = f'profile_{target}_32_20kcali_size_llama2.txt'
    # struct_spdy.search(profile)
    # db.load_file(module, profile)
    

    weight_path_ori = "/nfs/scistore19/alistgrp/stang/llm-foundry/scripts/database/llama_3_1_8b_evo_search_attn_prune_each_head_from_1.5x"
    weight_path = "/nfs/scistore19/alistgrp/stang/llm-foundry/scripts/database/llama_3_1_8b_evo_search_attn_prune_each_head_from_1.5x"
    # weight_path = "/nfs/scistore19/alistgrp/stang/llm-foundry/scripts/database/gradual_prune_from_1.5x"
    # weight_path = "/nfs/scistore19/alistgrp/stang/llm-foundry/scripts/database/2.5x_gradual_from_2_reg"
    # db_ = db.db
    # for name in db_:
    #     layer_dir = os.path.join(weight_path, name)
    #     if not os.path.exists(layer_dir):
    #         os.mkdir(layer_dir)
    #     for sparsity in list(db_[name].keys()):
    #         # sparsity_dir = os.path.join(layer_dir, sparsity)
    #         # if not os.path.exists(sparsity_dir):
    #         #     os.mkdir(sparsity_dir)
    #         torch.save(db_[name][sparsity], f"{layer_dir}/{sparsity}.pt")
    #     # sparsity_dir = os.path.join(layer_dir, "1.0000")
    #     # if not os.path.exists(sparsity_dir):
    #     #     os.mkdir(sparsity_dir)
    #     torch.save(torch.zeros_like(db_[name][sparsity]).cpu(), f"{layer_dir}/1.0000.pt")
    # return


    # errors = db.load_errors(error_file)
    # baselinetime, prunabletime, timings = db.get_berttimings(timings_file)
    
    # errors = db.load_errors(error_file)
    # baselinetime, prunabletime, timings = db.get_berttimings(timings_file)
    module.to("cuda:1")
    layer_names = []
    layer_list = os.listdir(weight_path_ori)
    for i in range(int(len(layer_list)/2)):
        layer_names.append(f"model.model.layers.{i}.self_attn.o_proj")
        layer_names.append(f"model.model.layers.{i}.mlp.down_proj")
    module.state = [None] * len(layer_names)
    # print(f"attn level2sparsity dict: {db.level2attn_sparsity}")
    # print(f"mlp level2sparsity dict: {db.level2mlp_sparsity}")
    # layer_names = []
    # for name in db.db:
    #     layer_names.append(name)
    #     print(name)
    
    
    
    # # 1.5x acceleration
    # parent = [3 for _ in layer_names]
    # 2x acceleration
    # parent = [3 for _ in layer_names]
    # For llama3.1
    parent = []
    for i, _ in enumerate(layer_names):
        if i % 2 == 0:
            #attn
            parent.append(5)
        else:
            #mlp
            parent.append(5)
    # 2.5x acceleration
    # parent = [6 for _ in layer_names]

    # 2.5 reg from 2.0x
    # new_parent = [3, 3, 0, 0, 5, 6, 4, 3, 4, 4, 5, 4, 4, 3, 3, 5, 3, 5, 4, 4, 5, 4, 4, 3, 5, 6, 4, 6, 7, 6, 8, 6, 1, 3, 4, 3, 5, 4, 6, 6, 7, 7, 7, 7, 6, 7, 8, 7, 5, 5, 6, 5, 3, 7, 10, 7, 7, 6, 8, 6, 5, 6, 4, 6]
    calib_loader = _dataloader_builder(dataloader,
                                    batchsize=1,
                                    nsamples=64, # Please adjust this number for efficiency
                                        )
    struct_evo_search = StructuredEvoSearch(model=module, db=None, weight_path=weight_path,
                                            calibration_dataloader=calib_loader, fitness_fn="ppl")
    struct_evo_search.get_berttimings(timings_file)
    
    # new_parent = []
    # new_parent = [0, 0, 0, 0, 2, 2, 1, 2, 2, 2, 2, 2, 3, 2, 1, 2, 5, 4, 4, 4, 3, 4, 4, 4, 3, 4, 2, 2, 4, 4, 3, 2, 1, 2, 2, 2, 2, 2, 3, 2, 4, 4, 3, 4, 4, 4, 5, 4, 3, 4, 4, 4, 5, 4, 4, 4, 4, 5, 5, 4, 3, 5, 4, 2]
    # for i, name in enumerate(layer_names):
    #     if os.path.exists(os.path.join(weight_path, name)):
    #         sparsity_list = os.listdir(os.path.join(weight_path, name))
    #         sparsities = []
    #         for sparsity in sparsity_list:
    #             sparsities.append(float(sparsity[:-3]))
    #         min_sparsities = min(sparsities)
    #         if min_sparsities == 0:
    #             new_parent.append(0)
    #             continue
    #         if i % 2 == 0:
    #             level = struct_evo_search.attnlevel2sparsity["%.4f" % min_sparsities]
    #         else:
    #             level = struct_evo_search.mlplevel2sparsity["%.4f" % min_sparsities]
    #         new_parent.append(int(level))
    #     else:
    #         new_parent.append(10)

    # module.state = copy.deepcopy(new_parent)
    
    # resdual = sum(parent) - sum(new_parent)
    # for _ in range(resdual):
    #     while True:
    #         idx = random.randint(4, len(parent) - 5)
    #         if new_parent[idx] + 1 > 9:
    #             continue
    #         else:
    #             new_parent[idx] += 1
    #             break
    
    # parent = new_parent


    # print("New parent: ", new_parent)
    # parent=[0, 3, 3, 6, 6, 3, 3, 0, 0, 5, 3, 1, 4, 3, 5, 3, 3, 5, 6, 8, 9, 3, 6, 8, 6, 6, 6, 6, 6, 6, 6, 6, 5, 8, 8, 8, 7, 6, 8, 6, 5, 10, 6, 7, 8, 4, 9, 7, 6, 10, 8, 9, 8, 5, 10, 8, 4, 10, 10, 8, 8, 6, 10, 8] 
    # manual= [0, 6, 2, 0, 4, 3, 6, 3, 2, 3, 5, 3, 3, 4, 6, 6, 6, 3, 3, 3, 4, 3, 1, 5, 6, 6, 1, 6, 7, 7, 8, 3, 3, 6, 4, 3, 5, 3, 6, 6, 3, 6, 4, 3, 5, 7, 6, 6, 6, 7, 6, 5, 4, 7, 6, 7, 6, 8, 9, 6, 6, 6, 5, 6]
    # manual = [3, 3, 0, 0, 5, 6, 4, 3, 4, 4, 5, 4, 4, 3, 3, 5, 3, 5, 4, 4, 5, 4, 4, 3, 5, 6, 4, 6, 7, 6, 8, 6, 1, 3, 4, 3, 5, 4, 6, 6, 7, 7, 7, 7, 6, 7, 8, 7, 5, 5, 6, 5, 3, 7, 10, 7, 7, 6, 8, 6, 5, 6, 4, 6]
    # manual = [0, 6, 1, 0, 4, 3, 4, 3, 4, 4, 5, 5, 6, 6, 5, 5, 3, 3, 3, 3, 4, 4, 4, 4, 5, 6, 5, 6, 7, 6, 7, 6, 4, 3, 4, 4, 5, 5, 5, 6, 5, 6, 6, 6, 5, 6, 8, 6, 5, 6, 6, 5, 5, 6, 8, 7, 6, 6, 6, 6, 6, 6, 6, 6]
    # manual = [5, 1, 2, 0, 5, 4, 3, 4, 5, 4, 5, 2, 3, 2, 3, 5, 6, 6, 6, 6, 5, 6, 5, 6, 5, 6, 4, 6, 5, 5, 5, 6, 4, 4, 5, 4, 4, 5, 4, 6, 8, 5, 7, 5, 8, 5, 8, 6, 5, 6, 5, 6, 6, 6, 6, 7, 6, 5, 6, 6, 5, 5, 7, 4]
    # manual = parent
    # print(manual)
    # manual = [0, 0, 0, 0, 2, 2, 1, 2, 2, 2, 2, 2, 3, 2, 1, 2, 5, 4, 4, 4, 3, 4, 4, 4, 3, 4, 2, 2, 4, 4, 3, 2, 1, 2, 2, 2, 2, 2, 3, 2, 4, 4, 3, 4, 4, 4, 5, 4, 3, 4, 4, 4, 5, 4, 4, 4, 4, 5, 5, 4, 3, 5, 4, 2]
    # manual = [0, 2, 0, 2, 3, 4, 2, 4, 4, 4, 5, 4, 3, 2, 3, 5, 6, 7, 5, 7, 5, 7, 5, 7, 5, 6, 5, 6, 7, 6, 5, 5, 3, 4, 5, 4, 5, 4, 4, 4, 7, 5, 5, 5, 7, 5, 6, 6, 5, 7, 5, 7, 7, 7, 5, 8, 6, 6, 7, 5, 7, 5, 8, 5]
    # manual[0] = 0
    # manual[1] = 0
    # manual[2] = 0
    # manual[3] = 0

    # manual[-1] = 0
    # manual[-2] = 0
    # manual[-3] = 0
    # manual[-4] = 0

    # manual[20] += 1 
    # manual[21] += 1
    # manual[22] += 1
    # manual[23] += 1

    # struct_evo_search.load_weight(module, layer_names, manual, weight_path)
    # print(struct_evo_search.compute_fitness(module, struct_evo_search.data[:40], "ppl", None))
    # return
    # print(struct_evo_search.compute_fitness(module, struct_evo_search.data[:64], "ppl", struct_evo_search.target_logits[:64]))
    # return
    # target_logits = []



    group_num = 0
    generation_number = 300
    within_group = False
    group_index = 0
    offspring_num = 16
    for generation in range(generation_number):
        # if generation < 20:
        #     group_num = 4
        # elif generation < 100:
        #     group_num = 8
        # elif generation < 300:
        offspring_num = 16
        group_num = 8 
        within_group = True
        group_index = (generation - 0) % group_num
        # elif generation < 200:
        #     offspring_num = 16
        #     group_num = 4
        #     within_group = True
        #     group_index = (generation - 100) % group_num
        # elif generation < 400:
        #     offspring_num = 32
        #     group_num = 4
        #     within_group = True
        #     group_index = (generation - 300) % group_num
        
        # elif generation < 150:
        #     group_num = 1
        #     offspring_num = 16
        #     within_group = True
        #     group_index=0
        
        # else:
        # group_num = 16
        # offspring_num = 8
        # within_group = True
        # group_index = 0
        # group_index = (generation - 0) % group_num

        print(f"Generation {generation + 1}/{generation_number}")
        if within_group:
            group_size = int(len(parent) / group_num)
            group_level = parent[group_size*group_index : group_size*group_index + group_size]
            attn_group_level = []
            mlp_group_level = []
            if all(i == 0 for i in group_level) or all(i == 10 for i in group_level):
                continue

            # for i in range(group_size):
            #     if i % 2 == 0:
            #         attn_group_level.append(group_level[i])
            #     else:
            #         mlp_group_level.append(group_level[i])

            # if all(i == 0 for i in attn_group_level) or all(i == 10 for i in attn_group_level):
            #     continue
            # if all(i == 0 for i in mlp_group_level) or all(i == 10 for i in mlp_group_level):
            #     continue
        print("Start to generate offspring.")
        offspring_list = struct_evo_search.generate_offspring(
            parent=parent, layer_names=layer_names, offspring_num=offspring_num, 
            group_num=group_num, within_group=within_group, group_index=group_index,
            max_level=10)
        print("Offspring generation is over. Ready to select.")

        parent, train_fitness = struct_evo_search.selection(
            model=module,
            parent=parent,
            offspring_list=offspring_list, 
            fitness_fn="ppl",
        layer_names=layer_names)
        print(f"Selection of generation {generation + 1} is over.")
        print(f"Best fitness value of generation {generation + 1} is {train_fitness}")
    profile = f'profile"_{target}_{run_name}.txt'
    with open(profile, "w") as f:
        f.write("\n".join([f"{layer_name}: {level}" for layer_name, level in zip(layer_names, parent)]))
    print("The final child is: ")
    print(parent)
    struct_evo_search.load_weight(module, layer_names, parent, weight_path)
    return

@torch.no_grad()
def load_pruned_model(module, db_file, profile):
    # db_file = f'database_20kcalib_32_size_llama_2_from1.5.db'
    
    # profile = f'profile_2_32_20kcali_size_llama2.txt'

    model = _get_model(module)()
    db = StructDatabase(db_file, model)
    
    db.load_file(module, profile)
    # shrink(module, update_mask=False)