"""Batched HuggingFace logit steering: measure how the correctness direction shifts logits.

Drop-in replacement for logit_steering_hf.run_logit_steering_hf with batch_size support.
"""

import gc
import random
from typing import Dict, List, Optional

import torch

from src.hf_utils import (
    ModelBundle,
    run_forward_hf,
    run_with_hooks_hf,
    get_letter_token_ids_hf,
)
from src.adaptive_utils import (
    get_answer_token_length_at_position,
    clip_offsets_for_token_length,
    verify_token_lengths,
)


def _extract_letter_logits(logits, letter_ids, seq_lengths):
    """Extract letter logits at the last real token for each item. Returns (B, 4) CPU float tensor."""
    B = logits.shape[0]
    letter_ids = letter_ids.to(logits.device)
    result = torch.zeros(B, len(letter_ids))
    for b in range(B):
        result[b] = logits[b, seq_lengths[b] - 1, letter_ids].float().cpu()
    return result


def run_logit_steering_hf(
    bundle: ModelBundle,
    data: Dict,
    direction_path: str,
    layers: List[int],
    offsets: List[int],
    scale: float,
    seed: int = 42,
    projections_path: str = None,
    alpha: float = 0.0,
    adaptive_offsets: bool = False,
    bias_correct: bool = False,
    batch_size: int = 16,
) -> Dict:
    """Batched logit steering: measure how the correctness direction shifts logits."""
    random.seed(seed)
    match_mean = projections_path is not None

    # Load direction
    direction_data = torch.load(direction_path, weights_only=False)
    direction = direction_data['directions']
    direction_offsets = direction_data['offsets']

    offset_indices = []
    for off in offsets:
        if off in direction_offsets:
            offset_indices.append(direction_offsets.index(off))
        else:
            raise ValueError(f"Offset {off} not in direction. Available: {direction_offsets}")

    direction = direction.to(bundle.device).to(bundle.dtype)

    # Load projection targets
    proj_targets = {}
    position_bias = {}
    if match_mean:
        import pickle
        with open(projections_path, 'rb') as f:
            proj_data = pickle.load(f)
        correct_mean = proj_data['correct_mean']
        incorrect_mean = proj_data['incorrect_mean']
        correct_std = proj_data['correct_std']
        incorrect_std = proj_data['incorrect_std']
        proj_offsets = list(proj_data['offsets'])
        proj_layers = list(proj_data['layers'])

        for oi, off in enumerate(offsets):
            pi = proj_offsets.index(off)
            for layer in layers:
                li = proj_layers.index(layer)
                c_mean = float(correct_mean[pi, li])
                i_mean = float(incorrect_mean[pi, li])
                c_std = float(correct_std[pi, li])
                i_std = float(incorrect_std[pi, li])
                proj_targets[(oi, layer)] = (
                    i_mean - alpha * i_std,
                    c_mean + alpha * c_std,
                )

        if bias_correct and 'position_means' in proj_data:
            pm = proj_data['position_means']
            for oi, off in enumerate(offsets):
                pi = proj_offsets.index(off)
                for layer in layers:
                    li = proj_layers.index(layer)
                    position_bias[(oi, layer)] = {
                        pos: float(pm[pos][pi, li]) for pos in pm
                    }
            print(f"  Bias correction enabled", flush=True)

        if alpha != 0.0:
            print(f"Match-mean+alpha mode: target = opposite_mean +/- {alpha} * std", flush=True)

    pairs = data['pairs']
    n_pairs = len(pairs)
    n_choices = max(len(p['delimiter_positions']) for p in pairs)
    letter_ids = get_letter_token_ids_hf(bundle.tokenizer, bundle.device, n_choices=n_choices)

    if adaptive_offsets:
        print("  Adaptive offsets enabled for logit steering", flush=True)
        bad = verify_token_lengths(bundle.tokenizer, pairs)
        if bad:
            pairs = [p for i, p in enumerate(pairs) if i not in bad]
            n_pairs = len(pairs)

    # Pre-assign targets
    correct_pos_list = []
    target_pos_list = []
    questions = []
    subjects = []
    answer_texts_list = []
    for pair in pairs:
        cp = pair['correct_pos']
        n_ch = len(pair['delimiter_positions'])
        incorrect_options = [i for i in range(n_ch) if i != cp]
        tp = random.choice(incorrect_options)
        correct_pos_list.append(cp)
        target_pos_list.append(tp)
        questions.append(pair.get('question', ''))
        subjects.append(pair.get('subject', ''))
        answer_texts_list.append(pair.get('answer_texts', []))

    # Output arrays
    baseline_lds = torch.zeros(n_pairs)
    steered_lds = torch.zeros(n_pairs)
    baseline_correct_logits = torch.zeros(n_pairs)
    baseline_target_logits = torch.zeros(n_pairs)
    baseline_other_max_logits = torch.zeros(n_pairs)
    steered_correct_logits = torch.zeros(n_pairs)
    steered_target_logits = torch.zeros(n_pairs)
    steered_other_max_logits = torch.zeros(n_pairs)
    n_skipped = 0

    pad_id = bundle.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = bundle.tokenizer.eos_token_id

    # Process in batches
    for chunk_start in range(0, n_pairs, batch_size):
        chunk_end = min(chunk_start + batch_size, n_pairs)
        B = chunk_end - chunk_start
        chunk_pairs = pairs[chunk_start:chunk_end]
        chunk_cp = correct_pos_list[chunk_start:chunk_end]
        chunk_tp = target_pos_list[chunk_start:chunk_end]
        chunk_delims = [list(p['delimiter_positions']) for p in chunk_pairs]

        # Tokenize and pad
        token_ids_list = [
            bundle.tokenizer.encode(p['gen'], add_special_tokens=False)
            for p in chunk_pairs
        ]
        seq_lengths = [len(t) for t in token_ids_list]
        max_len = max(seq_lengths)

        padded = torch.full((B, max_len), pad_id, dtype=torch.long, device=bundle.device)
        attn_mask = torch.zeros((B, max_len), dtype=torch.long, device=bundle.device)
        for b in range(B):
            padded[b, :seq_lengths[b]] = torch.tensor(token_ids_list[b], device=bundle.device)
            attn_mask[b, :seq_lengths[b]] = 1

        # Adaptive offsets per item
        if adaptive_offsets:
            chunk_valid_offsets = []
            for b in range(B):
                tl_c = get_answer_token_length_at_position(bundle.tokenizer, chunk_pairs[b], chunk_cp[b])
                tl_t = get_answer_token_length_at_position(bundle.tokenizer, chunk_pairs[b], chunk_tp[b])
                chunk_valid_offsets.append(set(clip_offsets_for_token_length(offsets, min(tl_c, tl_t))))
        else:
            chunk_valid_offsets = [set(offsets)] * B

        # Check validity: all positions in bounds
        skip_mask = [False] * B
        for b in range(B):
            for oi, offset in enumerate(offsets):
                if offset not in chunk_valid_offsets[b]:
                    continue
                pc = chunk_delims[b][chunk_cp[b]] + offset
                pt = chunk_delims[b][chunk_tp[b]] + offset
                if pc < 0 or pc >= seq_lengths[b] or pt < 0 or pt >= seq_lengths[b]:
                    skip_mask[b] = True
                    break

        try:
            with torch.no_grad():
                # Baseline forward — extract letter logits immediately to free full logits
                raw_logits = run_forward_hf(bundle, padded, attention_mask=attn_mask)
                baseline_letters = _extract_letter_logits(raw_logits, letter_ids, seq_lengths)
                del raw_logits

            # Build batched hooks
            # Each hook must handle all items in the batch, with per-item interventions
            fwd_hooks = []
            for layer in layers:
                interventions = []  # list of (batch_idx, type, ...)
                for b in range(B):
                    if skip_mask[b]:
                        continue
                    for oi, offset in enumerate(offsets):
                        if offset not in chunk_valid_offsets[b]:
                            continue
                        pc = chunk_delims[b][chunk_cp[b]] + offset
                        pt = chunk_delims[b][chunk_tp[b]] + offset
                        dir_vec = direction[offset_indices[oi], layer]

                        if match_mean:
                            ct = proj_targets[(oi, layer)][0]
                            tt = proj_targets[(oi, layer)][1]
                            if bias_correct and (oi, layer) in position_bias:
                                pb = position_bias[(oi, layer)]
                                ct = ct + pb[chunk_cp[b]]
                                tt = tt + pb[chunk_tp[b]]
                            interventions.append(('match', b, pc, pt, dir_vec, ct, tt))
                        else:
                            interventions.append(('fixed', b, pc, pt, dir_vec, scale))

                if interventions:
                    def make_hook(intervs):
                        def hook_fn(activation):
                            for interv in intervs:
                                if interv[0] == 'match':
                                    _, b, pc, pt, dv, ct, tt = interv
                                    dv = dv.to(activation.device)
                                    proj_c = (activation[b, pc] * dv).sum()
                                    proj_t = (activation[b, pt] * dv).sum()
                                    activation[b, pc] = activation[b, pc] + (ct - proj_c) * dv
                                    activation[b, pt] = activation[b, pt] + (tt - proj_t) * dv
                                else:
                                    _, b, pc, pt, dv, s = interv
                                    dv = dv.to(activation.device)
                                    activation[b, pc] = activation[b, pc] - s * dv
                                    activation[b, pt] = activation[b, pt] + s * dv
                            return activation
                        return hook_fn
                    fwd_hooks.append((layer, make_hook(interventions)))

            with torch.no_grad():
                raw_logits = run_with_hooks_hf(
                    bundle, padded, fwd_hooks=fwd_hooks, attention_mask=attn_mask)
                steered_letters = _extract_letter_logits(raw_logits, letter_ids, seq_lengths)
                del raw_logits

            # Store results
            for b in range(B):
                pi = chunk_start + b
                if skip_mask[b]:
                    n_skipped += 1
                    for arr in [baseline_lds, steered_lds,
                                baseline_correct_logits, baseline_target_logits,
                                baseline_other_max_logits,
                                steered_correct_logits, steered_target_logits,
                                steered_other_max_logits]:
                        arr[pi] = float('nan')
                    continue

                cp, tp = chunk_cp[b], chunk_tp[b]
                baseline_lds[pi] = baseline_letters[b][cp] - baseline_letters[b][tp]
                steered_lds[pi] = steered_letters[b][cp] - steered_letters[b][tp]

                baseline_correct_logits[pi] = baseline_letters[b][cp]
                baseline_target_logits[pi] = baseline_letters[b][tp]
                steered_correct_logits[pi] = steered_letters[b][cp]
                steered_target_logits[pi] = steered_letters[b][tp]

                n_ch_b = len(chunk_pairs[b]['delimiter_positions'])
                other_idx = [i for i in range(n_ch_b) if i != cp and i != tp]
                if other_idx:
                    baseline_other_max_logits[pi] = baseline_letters[b][other_idx].max()
                    steered_other_max_logits[pi] = steered_letters[b][other_idx].max()
                else:
                    baseline_other_max_logits[pi] = float('nan')
                    steered_other_max_logits[pi] = float('nan')

        except Exception as e:
            print(f"  Error in batch {chunk_start}: {e}", flush=True)
            for b in range(B):
                pi = chunk_start + b
                for arr in [baseline_lds, steered_lds,
                            baseline_correct_logits, baseline_target_logits,
                            baseline_other_max_logits,
                            steered_correct_logits, steered_target_logits,
                            steered_other_max_logits]:
                    arr[pi] = float('nan')
                n_skipped += 1

        gc.collect()
        torch.cuda.empty_cache()

        processed = chunk_end
        if processed % (batch_size * 10) == 0 or processed == n_pairs:
            valid_so_far = ~torch.isnan(steered_lds[:processed])
            mean_delta = (baseline_lds[:processed] - steered_lds[:processed])[valid_so_far].mean().item()
            print(f"  Processed {processed}/{n_pairs} pairs (skipped {n_skipped}), "
                  f"running mean delta_LD: {mean_delta:.4f}", flush=True)

    # Compute delta (negative = steering reduced correct answer's advantage)
    delta_lds = steered_lds - baseline_lds
    valid = ~torch.isnan(delta_lds)

    mean_baseline = baseline_lds[valid].mean().item()
    mean_steered = steered_lds[valid].mean().item()
    mean_delta = delta_lds[valid].mean().item()

    print(f"\nLogit steering complete (HF batched): {n_pairs} pairs, {n_skipped} skipped",
          flush=True)
    print(f"Mean baseline LD: {mean_baseline:.4f}", flush=True)
    print(f"Mean steered LD:  {mean_steered:.4f}", flush=True)
    print(f"Mean delta LD:    {mean_delta:.4f}", flush=True)

    print(f"\nIndividual logit breakdown:", flush=True)
    for name, bl, st in [
        ('Correct answer', baseline_correct_logits, steered_correct_logits),
        ('Target answer', baseline_target_logits, steered_target_logits),
        ('Highest other', baseline_other_max_logits, steered_other_max_logits),
    ]:
        bl_mean = bl[valid].mean().item()
        st_mean = st[valid].mean().item()
        print(f"  {name:20s}: baseline={bl_mean:.4f}, steered={st_mean:.4f}, "
              f"delta={st_mean - bl_mean:.4f}", flush=True)

    return {
        'baseline_ld': baseline_lds,
        'steered_ld': steered_lds,
        'delta_ld': delta_lds,
        'baseline_correct_logits': baseline_correct_logits,
        'baseline_target_logits': baseline_target_logits,
        'baseline_other_max_logits': baseline_other_max_logits,
        'steered_correct_logits': steered_correct_logits,
        'steered_target_logits': steered_target_logits,
        'steered_other_max_logits': steered_other_max_logits,
        'correct_pos': correct_pos_list,
        'target_pos': target_pos_list,
        'questions': questions,
        'subjects': subjects,
        'answer_texts': answer_texts_list,
        'metadata': {
            'n_pairs': n_pairs,
            'n_skipped': n_skipped,
            'layers': layers,
            'offsets': offsets,
            'scale': scale,
            'seed': seed,
            'match_mean': match_mean,
            'alpha': alpha,
            'adaptive_offsets': adaptive_offsets,
            'batch_size': batch_size,
            'backend': 'huggingface_batched',
            'mean_baseline_ld': mean_baseline,
            'mean_steered_ld': mean_steered,
            'mean_delta_ld': mean_delta,
            'mean_baseline_correct_logit': baseline_correct_logits[valid].mean().item(),
            'mean_baseline_target_logit': baseline_target_logits[valid].mean().item(),
            'mean_baseline_other_max_logit': baseline_other_max_logits[valid].mean().item(),
            'mean_steered_correct_logit': steered_correct_logits[valid].mean().item(),
            'mean_steered_target_logit': steered_target_logits[valid].mean().item(),
            'mean_steered_other_max_logit': steered_other_max_logits[valid].mean().item(),
        },
    }
