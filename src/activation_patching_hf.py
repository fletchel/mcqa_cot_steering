"""HuggingFace version of activation_patching.py: random-swap patching with HF models.

Drop-in replacement using HF register_forward_hook instead of TransformerLens
run_with_cache / run_with_hooks. Supports quantization and multi-GPU.

Plotting functions are reused from activation_patching.py (they operate on tensors only).
"""

import gc
import os
import random
from typing import List, Dict, Optional

import torch
import numpy as np

from src.hf_utils import (
    ModelBundle,
    run_with_cache_hf,
    run_with_hooks_hf,
    get_letter_token_ids_hf,
    tokenize_and_pad,
)
from src.adaptive_utils import (
    get_answer_token_length_at_position,
    clip_offsets_for_token_length,
    get_min_offsets,
    group_pairs_by_token_length,
    verify_token_lengths,
    save_debug_file,
)

# Reuse plotting functions from original module
from src.activation_patching import (
    plot_patching_heatmap,
    plot_patching_lineplot,
    plot_patching_lineplot_by_token_length,
    plot_patching_heatmaps_by_token_length,
    get_example_tokens_at_offsets,
)


def _get_logit_diffs_batched(
    logits: torch.Tensor,
    letter_ids: torch.Tensor,
    seq_lengths: List[int],
    pos1_list: List[int],
    pos2_list: List[int],
) -> List[float]:
    """Compute logit diffs for a padded batch, using each item's actual last token."""
    diffs = []
    for b in range(len(seq_lengths)):
        last_tok = seq_lengths[b] - 1
        final_logits = logits[b, last_tok, letter_ids]
        diffs.append((final_logits[pos1_list[b]] - final_logits[pos2_list[b]]).item())
    return diffs


def run_random_swap_patching_hf(
    bundle: ModelBundle,
    data: Dict,
    offsets: List[int],
    layers: Optional[List[int]] = None,
    seed: int = 42,
    window_size: int = 0,
    all_offsets: bool = False,
    batch_size: int = 4,
    adaptive_offsets: bool = False,
    max_pairs: Optional[int] = None,
    both: bool = False,
) -> Dict:
    """Random swap patching — HF version.

    Identical interface and output format to activation_patching.run_random_swap_patching.
    """
    random.seed(seed)
    pairs = data['pairs']

    if layers is None:
        layers = list(range(bundle.n_layers))
    n_layers = len(layers)
    n_offsets = len(offsets)

    max_n_choices = max(len(p['delimiter_positions']) for p in pairs)
    letter_ids = get_letter_token_ids_hf(bundle.tokenizer, bundle.device,
                                         n_choices=max_n_choices)
    n_skipped = 0

    # Layers to cache (including window)
    cache_layers = set(layers)
    for l in layers:
        for w in range(-window_size, window_size + 1):
            if 0 <= l + w < bundle.n_layers:
                cache_layers.add(l + w)
    cache_layers = sorted(cache_layers)

    # Compute per-answer token lengths
    tokenizer = bundle.tokenizer
    answer_token_lengths = []
    for pair in pairs:
        lengths = tuple(
            len(tokenizer.encode(pair['answer_texts'][pair['order'][i]],
                                 add_special_tokens=False))
            for i in range(len(pair['order']))
        )
        answer_token_lengths.append(lengths)

    # Pre-assign random partners (same logic as original)
    all_pairs = data['pairs']
    all_lengths = answer_token_lengths

    matched_pairs = []
    for i in range(len(all_pairs)):
        my_pos = all_pairs[i]['correct_pos']
        my_lengths = all_lengths[i]
        candidates = [
            j for j in range(len(all_pairs))
            if j != i
            and len(all_lengths[j]) == len(my_lengths)  # same n_choices
            and all_pairs[j]['correct_pos'] != my_pos
            and all_lengths[j][my_pos] == my_lengths[my_pos]
            and all_lengths[j][all_pairs[j]['correct_pos']] == my_lengths[all_pairs[j]['correct_pos']]
        ]
        if candidates:
            matched_pairs.append((i, random.choice(candidates)))

    n_total_matched = len(matched_pairs)
    if max_pairs and len(matched_pairs) > max_pairs:
        random.shuffle(matched_pairs)
        matched_pairs = matched_pairs[:max_pairs]

    selected_indices = [m[0] for m in matched_pairs]
    pairs = [all_pairs[i] for i in selected_indices]
    data = {**data, 'pairs': pairs}
    n_pairs = len(pairs)
    partner_indices = [m[1] for m in matched_pairs]
    answer_token_lengths = [all_lengths[i] for i in selected_indices]

    print(f"  Random swap partners: {n_total_matched}/{len(all_pairs)} pairs have token-matched partners", flush=True)
    print(f"  Using {n_pairs} pairs for random swap patching", flush=True)

    do_per_offset = not all_offsets or both
    do_all_offsets = all_offsets or both

    metrics_per_offset = torch.zeros(n_pairs, n_layers, n_offsets) if do_per_offset else None
    metrics_all_offsets = torch.zeros(n_pairs, n_layers) if do_all_offsets else None

    processed = 0
    for chunk_start in range(0, n_pairs, batch_size):
        chunk_end = min(chunk_start + batch_size, n_pairs)
        chunk_indices = list(range(chunk_start, chunk_end))
        B = len(chunk_indices)

        chunk_pairs = [pairs[i] for i in chunk_indices]
        chunk_partners = [all_pairs[partner_indices[i]] for i in chunk_indices]

        letter_pos1_list = [p['correct_pos'] for p in chunk_pairs]
        letter_pos2_list = [q['correct_pos'] for q in chunk_partners]
        delim1_list = [list(p['delimiter_positions']) for p in chunk_pairs]
        delim2_list = [list(q['delimiter_positions']) for q in chunk_partners]
        patch_letter_positions_list = [
            [lp2, lp1] for lp1, lp2 in zip(letter_pos1_list, letter_pos2_list)
        ]

        # Tokenize and pad
        q1_batch = tokenize_and_pad(
            tokenizer, [p['gen'] for p in chunk_pairs], bundle.device)
        q2_batch = tokenize_and_pad(
            tokenizer, [q['gen'] for q in chunk_partners], bundle.device)

        q1_lengths = q1_batch['lengths']
        q2_lengths = q2_batch['lengths']

        try:
            # Cache forward passes
            baseline_logits, cache1 = run_with_cache_hf(
                bundle, q1_batch['input_ids'],
                attention_mask=q1_batch['attention_mask'],
                layers=cache_layers,
            )
            corrupt_logits, cache2 = run_with_cache_hf(
                bundle, q2_batch['input_ids'],
                attention_mask=q2_batch['attention_mask'],
                layers=cache_layers,
            )

            baseline_diffs = _get_logit_diffs_batched(
                baseline_logits, letter_ids, q1_lengths,
                letter_pos1_list, letter_pos2_list,
            )
            corrupt_diffs = _get_logit_diffs_batched(
                corrupt_logits, letter_ids, q2_lengths,
                letter_pos1_list, letter_pos2_list,
            )

            denominators = [bd - cd for bd, cd in zip(baseline_diffs, corrupt_diffs)]

            valid_mask = [abs(d) >= 1e-6 for d in denominators]
            for b in range(B):
                if not valid_mask[b]:
                    n_skipped += 1
                    pi = chunk_start + b
                    if metrics_per_offset is not None:
                        metrics_per_offset[pi] = float('nan')
                    if metrics_all_offsets is not None:
                        metrics_all_offsets[pi] = float('nan')

            # Adaptive offsets per chunk item
            if adaptive_offsets:
                chunk_valid_offsets = []
                chunk_offsets_per_pos = []
                for b in range(B):
                    offsets_per_pos = {}
                    for lp in patch_letter_positions_list[b]:
                        tl1 = get_answer_token_length_at_position(tokenizer, chunk_pairs[b], lp)
                        tl2 = get_answer_token_length_at_position(tokenizer, chunk_partners[b], lp)
                        offsets_per_pos[lp] = clip_offsets_for_token_length(offsets, min(tl1, tl2))
                    valid_set = set(offsets)
                    for lp in patch_letter_positions_list[b]:
                        valid_set &= set(offsets_per_pos[lp])
                    chunk_valid_offsets.append(valid_set)
                    chunk_offsets_per_pos.append(offsets_per_pos)

            for layer_i, layer in enumerate(layers):

                # All-offsets mode
                if do_all_offsets:
                    skip_items = set()
                    source_acts = {}

                    for b in range(B):
                        if not valid_mask[b]:
                            skip_items.add(b)
                            continue
                        if adaptive_offsets:
                            b_offsets = set()
                            for lp in patch_letter_positions_list[b]:
                                b_offsets |= set(chunk_offsets_per_pos[b][lp])
                        else:
                            b_offsets = set(offsets)

                        for w in range(-window_size, window_size + 1):
                            wl = layer + w
                            if wl < 0 or wl >= bundle.n_layers:
                                skip_items.add(b)
                                break
                        if b in skip_items:
                            continue

                        for offset in offsets:
                            for lp in patch_letter_positions_list[b]:
                                if adaptive_offsets and offset not in chunk_offsets_per_pos[b][lp]:
                                    continue
                                pos1 = delim1_list[b][lp] + offset
                                pos2 = delim2_list[b][lp] + offset
                                if (pos1 < 0 or pos1 >= q1_lengths[b] or
                                        pos2 < 0 or pos2 >= q2_lengths[b]):
                                    skip_items.add(b)
                                    break
                                for w in range(-window_size, window_size + 1):
                                    wl = layer + w
                                    # cache2[wl] is (B, seq_len, d_model) on CPU
                                    source_acts[(b, lp, offset, wl)] = cache2[wl][b, pos2].clone()
                            if b in skip_items:
                                break

                    # Build hooks
                    fwd_hooks = []
                    for w in range(-window_size, window_size + 1):
                        wl = layer + w
                        if wl < 0 or wl >= bundle.n_layers:
                            continue

                        def make_alloff_hook(wl_val, skip_set, src_acts, dm1, plp_list,
                                             offs, adpt, opp_list):
                            def hook_fn(activation):
                                for b in range(activation.shape[0]):
                                    if b in skip_set:
                                        continue
                                    for offset in offs:
                                        for lp in plp_list[b]:
                                            if adpt and offset not in opp_list[b][lp]:
                                                continue
                                            key = (b, lp, offset, wl_val)
                                            if key in src_acts:
                                                pos = dm1[b][lp] + offset
                                                activation[b, pos] = src_acts[key].to(activation.device)
                                return activation
                            return hook_fn

                        fwd_hooks.append((wl, make_alloff_hook(
                            wl, skip_items, source_acts, delim1_list,
                            patch_letter_positions_list, offsets,
                            adaptive_offsets,
                            chunk_offsets_per_pos if adaptive_offsets else [None] * B,
                        )))

                    patched_logits = run_with_hooks_hf(
                        bundle, q1_batch['input_ids'],
                        attention_mask=q1_batch['attention_mask'],
                        fwd_hooks=fwd_hooks,
                    )
                    patched_diffs = _get_logit_diffs_batched(
                        patched_logits, letter_ids, q1_lengths,
                        letter_pos1_list, letter_pos2_list,
                    )
                    for b in range(B):
                        pi = chunk_start + b
                        if b in skip_items:
                            metrics_all_offsets[pi, layer_i] = float('nan')
                        else:
                            metrics_all_offsets[pi, layer_i] = (
                                (baseline_diffs[b] - patched_diffs[b]) / denominators[b]
                            )

                # Per-offset mode
                if do_per_offset:
                    for offset_i, offset in enumerate(offsets):
                        skip_items = set()
                        source_acts = {}

                        for b in range(B):
                            if not valid_mask[b]:
                                skip_items.add(b)
                                continue
                            if adaptive_offsets and offset not in chunk_valid_offsets[b]:
                                skip_items.add(b)
                                continue

                            for w in range(-window_size, window_size + 1):
                                wl = layer + w
                                if wl < 0 or wl >= bundle.n_layers:
                                    skip_items.add(b)
                                    break
                            if b in skip_items:
                                continue

                            for lp in patch_letter_positions_list[b]:
                                pos1 = delim1_list[b][lp] + offset
                                pos2 = delim2_list[b][lp] + offset
                                if (pos1 < 0 or pos1 >= q1_lengths[b] or
                                        pos2 < 0 or pos2 >= q2_lengths[b]):
                                    skip_items.add(b)
                                    break
                                for w in range(-window_size, window_size + 1):
                                    wl = layer + w
                                    source_acts[(b, lp, wl)] = cache2[wl][b, pos2].clone()

                        if len(skip_items) == B:
                            for b in range(B):
                                metrics_per_offset[chunk_start + b, layer_i, offset_i] = float('nan')
                            continue

                        fwd_hooks = []
                        for w in range(-window_size, window_size + 1):
                            wl = layer + w
                            if wl < 0 or wl >= bundle.n_layers:
                                continue

                            def make_peroff_hook(wl_val, off, skip_set, src_acts, dm1, plp_list):
                                def hook_fn(activation):
                                    for b in range(activation.shape[0]):
                                        if b in skip_set:
                                            continue
                                        for lp in plp_list[b]:
                                            key = (b, lp, wl_val)
                                            if key in src_acts:
                                                pos = dm1[b][lp] + off
                                                activation[b, pos] = src_acts[key].to(activation.device)
                                    return activation
                                return hook_fn

                            fwd_hooks.append((wl, make_peroff_hook(
                                wl, offset, skip_items, source_acts,
                                delim1_list, patch_letter_positions_list,
                            )))

                        patched_logits = run_with_hooks_hf(
                            bundle, q1_batch['input_ids'],
                            attention_mask=q1_batch['attention_mask'],
                            fwd_hooks=fwd_hooks,
                        )
                        patched_diffs = _get_logit_diffs_batched(
                            patched_logits, letter_ids, q1_lengths,
                            letter_pos1_list, letter_pos2_list,
                        )
                        for b in range(B):
                            pi = chunk_start + b
                            if b in skip_items:
                                metrics_per_offset[pi, layer_i, offset_i] = float('nan')
                            else:
                                metrics_per_offset[pi, layer_i, offset_i] = (
                                    (baseline_diffs[b] - patched_diffs[b]) / denominators[b]
                                )

        except Exception as e:
            print(f"  Error on chunk {chunk_start}-{chunk_end}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            for b in range(B):
                n_skipped += 1
                pi = chunk_start + b
                if metrics_per_offset is not None:
                    metrics_per_offset[pi] = float('nan')
                if metrics_all_offsets is not None:
                    metrics_all_offsets[pi] = float('nan')

        # Clean up caches
        try:
            del cache1, cache2, baseline_logits, corrupt_logits
        except NameError:
            pass
        gc.collect()
        torch.cuda.empty_cache()

        processed = chunk_end
        _rm_tensor = metrics_per_offset if metrics_per_offset is not None else metrics_all_offsets
        valid = ~torch.isnan(_rm_tensor[:processed])
        running_mean = torch.nanmean(_rm_tensor[:processed]).item() if valid.any() else float('nan')
        print(f"  Processed {processed}/{n_pairs} pairs "
              f"(skipped {n_skipped}), running mean: {running_mean:.4f}", flush=True)

    print(f"\nRandom swap patching complete (HF): {n_pairs} pairs, {n_skipped} skipped", flush=True)

    if both:
        mean_per_offset = torch.nanmean(metrics_per_offset, dim=0)
        mean_all_offsets = torch.nanmean(metrics_all_offsets, dim=0)
        print(f"Mean metric (per-offset): {torch.nanmean(mean_per_offset).item():.4f}", flush=True)
        print(f"Mean metric (all-offsets): {torch.nanmean(mean_all_offsets).item():.4f}", flush=True)
        metrics = metrics_per_offset
        mean_metrics = mean_per_offset
    elif all_offsets:
        metrics = metrics_all_offsets
        mean_metrics = torch.nanmean(metrics, dim=0)
        print(f"Mean metric: {torch.nanmean(mean_metrics).item():.4f}", flush=True)
    else:
        metrics = metrics_per_offset
        mean_metrics = torch.nanmean(metrics, dim=0)
        print(f"Mean metric: {torch.nanmean(mean_metrics).item():.4f}", flush=True)

    tok_len_counts = {k: len(v) for k, v in group_pairs_by_token_length(tokenizer, pairs).items()}

    result = {
        'metrics': metrics,
        'mean_metrics': mean_metrics,
        'layers': layers,
        'offsets': offsets,
        'pairs': pairs,
        'metadata': {
            'n_pairs': n_pairs,
            'n_skipped': n_skipped,
            'n_layers': n_layers,
            'n_offsets': n_offsets,
            'patch_mode': 'random_swap',
            'window_size': window_size,
            'all_offsets': all_offsets,
            'adaptive_offsets': adaptive_offsets,
            'both': both,
            'backend': 'huggingface',
            'token_length_counts': tok_len_counts,
        },
    }

    if both:
        result['alloff_metrics'] = metrics_all_offsets
        result['alloff_mean_metrics'] = mean_all_offsets

    return result
