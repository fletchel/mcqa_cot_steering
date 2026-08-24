"""Memory-efficient direction finding: two-pass streaming approach.

Pass 1: Compute direction via running sums (O(1) memory in n_samples).
Pass 2: Project each residual onto direction, store only scalar projections.

Replaces extract_residual_at_delimiters_hf + compute_correct_direction +
compute_direction_projections with a single function that never stores
all residual vectors simultaneously.
"""

import gc
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm.auto import tqdm

import os
from src.utils import get_prompt_answer_positions, save_directions, summarize_projections

from src.hf_utils import (
    ModelBundle, load_model_hf,
    run_with_cache_hf, run_forward_hf,
    get_letter_token_ids_hf,
)
from src.direction_finder import (
    model_slug, strip_trailing_special_tokens, strip_cot_hint,
    _convert_patching_pairs, _filter_negated,
    plot_projection_separation,
)


def _iterate_residuals(bundle, data, offsets, layers, adaptive_offsets, verbose=True):
    """Yield (item_index, answer_idx, is_correct, resid_at_offsets, logits) for each answer.

    resid_at_offsets: (n_offsets, n_layers, d_model) tensor, may contain NaN.
    Only yields answers with at least one valid offset.
    """
    max_n_choices = max(len(item['pair'][0]['shuffled_answers']) for item in data)
    letter_ids = get_letter_token_ids_hf(bundle.tokenizer, bundle.device, n_choices=max_n_choices)
    n_offsets = len(offsets)
    n_layers = len(layers)
    d_model = bundle.d_model

    iterator = tqdm(data, desc="Processing") if verbose else data
    skipped = 0

    for item_idx, item in enumerate(iterator):
        clean = item['pair'][0]
        gen = clean['gen']
        choices = clean['shuffled_answers']
        correct_pos = clean['prompt_pos']

        pos_info = get_prompt_answer_positions(bundle.tokenizer, gen, choices)
        if not pos_info['success']:
            skipped += 1
            continue

        delimiter_positions = list(pos_info['positions'])

        input_ids = torch.tensor(
            bundle.tokenizer.encode(gen, add_special_tokens=False),
            device=bundle.device,
        ).unsqueeze(0)

        logits, cache = run_with_cache_hf(bundle, input_ids, layers=layers)
        n_ch = len(choices)
        final_letter_logits = logits[0, -1, letter_ids[:n_ch]].float().cpu()
        seq_len = cache[layers[0]].shape[1]

        for answer_idx, delim_pos in enumerate(delimiter_positions):
            is_correct = (answer_idx == correct_pos)

            if adaptive_offsets:
                answer_text = choices[answer_idx]
                token_length = len(bundle.tokenizer.encode(
                    " " + answer_text, add_special_tokens=False))
                min_valid_offset = -token_length

            resid = torch.full((n_offsets, n_layers, d_model), float('nan'))
            any_valid = False

            for oi, offset in enumerate(offsets):
                if adaptive_offsets and offset < min_valid_offset:
                    continue
                pos = delim_pos + offset
                if pos < 0 or pos >= seq_len:
                    if not adaptive_offsets:
                        any_valid = False
                        break
                    continue
                any_valid = True
                for li, layer in enumerate(layers):
                    resid[oi, li] = cache[layer][0, pos]

            if any_valid:
                yield item_idx, answer_idx, is_correct, resid, final_letter_logits

        del cache
        gc.collect()
        torch.cuda.empty_cache()

    if verbose:
        print(f"\nSkipped {skipped}/{len(data)} items (delimiter parse failures)", flush=True)


def extract_direction_and_projections_streaming(
    bundle: ModelBundle,
    data: List[Dict],
    offsets: List[int],
    layers: Optional[List[int]] = None,
    adaptive_offsets: bool = False,
    existing_direction: Optional[torch.Tensor] = None,
    verbose: bool = True,
) -> tuple:
    """Two-pass streaming extraction of direction and projections.

    Pass 1: Compute direction = mean(correct) - mean(incorrect) via running sums.
    Pass 2: Project each residual onto direction, store scalar projections.

    Returns (direction, proj_result) where:
        direction: (n_offsets, n_layers, d_model)
        proj_result: dict with correct_projections, incorrect_projections, means, stds, etc.
    """
    if layers is None:
        layers = list(range(bundle.n_layers))

    n_offsets = len(offsets)
    n_layers = len(layers)
    d_model = bundle.d_model

    # ---- Pass 1: Compute direction via running sums ----
    if existing_direction is not None:
        direction = existing_direction
        if verbose:
            print(f"Using existing direction: {direction.shape}", flush=True)
    else:
        if verbose:
            print("Pass 1: Computing direction via running sums...", flush=True)

        # Running sums — use float64 for numerical stability
        correct_sum = torch.zeros(n_offsets, n_layers, d_model, dtype=torch.float64)
        incorrect_sum = torch.zeros(n_offsets, n_layers, d_model, dtype=torch.float64)
        correct_count = torch.zeros(n_offsets, n_layers, dtype=torch.float64)
        incorrect_count = torch.zeros(n_offsets, n_layers, dtype=torch.float64)

        for item_idx, answer_idx, is_correct, resid, _ in _iterate_residuals(
                bundle, data, offsets, layers, adaptive_offsets, verbose):
            resid64 = resid.double()
            valid_mask = ~torch.isnan(resid64[:, :, 0])  # (n_offsets, n_layers)

            if is_correct:
                correct_sum += torch.where(valid_mask.unsqueeze(-1), resid64, torch.zeros_like(resid64))
                correct_count += valid_mask.double()
            else:
                incorrect_sum += torch.where(valid_mask.unsqueeze(-1), resid64, torch.zeros_like(resid64))
                incorrect_count += valid_mask.double()

        correct_mean = correct_sum / correct_count.unsqueeze(-1).clamp(min=1)
        incorrect_mean = incorrect_sum / incorrect_count.unsqueeze(-1).clamp(min=1)
        direction = (correct_mean - incorrect_mean).float()

        # Normalize
        norms = direction.norm(dim=-1, keepdim=True)
        direction = direction / (norms + 1e-8)

        n_correct = int(correct_count.max().item())
        n_incorrect = int(incorrect_count.max().item())
        if verbose:
            print(f"  Collected {n_correct} correct, {n_incorrect} incorrect", flush=True)
            if adaptive_offsets:
                for oi, offset in enumerate(offsets):
                    total = int(correct_count[oi, 0].item() + incorrect_count[oi, 0].item())
                    print(f"  Offset {offset:+d}: {total} valid answers", flush=True)

    # ---- Pass 2: Compute scalar projections ----
    if verbose:
        print("\nPass 2: Computing projections...", flush=True)

    dir_normalized = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

    # Collect projections and metadata
    correct_projections = []
    incorrect_projections = []  # flat list of all incorrect projections
    incorrect_question_idx = []  # which question each incorrect projection belongs to
    per_question_logits = []
    correct_positions = []
    questions = []
    answer_choices = []
    n_choices_per_question = []  # track variable n_choices

    # Track which questions we've seen (for logits — one per question)
    seen_questions = set()
    question_order = []  # item_idx in order of first appearance
    current_q_n_inc = {}  # item_idx -> count of incorrect answers seen

    for item_idx, answer_idx, is_correct, resid, letter_logits in _iterate_residuals(
            bundle, data, offsets, layers, adaptive_offsets, verbose):

        # Project: dot product along d_model
        proj = (resid * dir_normalized).sum(dim=-1)  # (n_offsets, n_layers)

        if is_correct:
            correct_projections.append(proj)
        else:
            incorrect_projections.append(proj)
            q_order_idx = question_order.index(item_idx) if item_idx in seen_questions else len(question_order)
            incorrect_question_idx.append(q_order_idx)

        # Store per-question metadata (once per question)
        if item_idx not in seen_questions:
            seen_questions.add(item_idx)
            question_order.append(item_idx)
            per_question_logits.append(letter_logits)
            clean = data[item_idx]['pair'][0]
            correct_positions.append(clean['prompt_pos'])
            questions.append(data[item_idx].get('question', ''))
            answer_choices.append(clean['shuffled_answers'])
            n_choices_per_question.append(len(clean['shuffled_answers']))

    # Stack
    correct_proj_tensor = torch.stack(correct_projections) if correct_projections else torch.zeros(0, n_offsets, n_layers)
    incorrect_proj_tensor = torch.stack(incorrect_projections) if incorrect_projections else torch.zeros(0, n_offsets, n_layers)
    if per_question_logits:
        max_logits = max(l.shape[0] for l in per_question_logits)
        padded_logits = []
        for l in per_question_logits:
            if l.shape[0] < max_logits:
                pad = torch.full((max_logits - l.shape[0],), float('nan'))
                padded_logits.append(torch.cat([l, pad]))
            else:
                padded_logits.append(l)
        logits_tensor = torch.stack(padded_logits)
    else:
        logits_tensor = torch.zeros(0, 4)

    # Statistics
    def _nanstd(t, dim):
        mask = ~torch.isnan(t)
        count = mask.sum(dim=dim).clamp(min=1)
        mean = torch.nanmean(t, dim=dim)
        diff = torch.where(mask, t - mean.unsqueeze(dim), torch.zeros_like(t))
        return (diff.pow(2).sum(dim=dim) / count.clamp(min=1)).sqrt()

    correct_mean_proj = torch.nanmean(correct_proj_tensor, dim=0)
    incorrect_mean_proj = torch.nanmean(incorrect_proj_tensor, dim=0)

    proj_result = {
        'correct_projections': correct_proj_tensor,
        'incorrect_projections': incorrect_proj_tensor,
        'incorrect_question_idx': incorrect_question_idx,
        'n_choices_per_question': n_choices_per_question,
        'correct_mean': correct_mean_proj,
        'incorrect_mean': incorrect_mean_proj,
        'correct_std': _nanstd(correct_proj_tensor, dim=0),
        'incorrect_std': _nanstd(incorrect_proj_tensor, dim=0),
        'separation': correct_mean_proj - incorrect_mean_proj,
        'offsets': offsets,
        'layers': layers,
        'letter_logits': logits_tensor,
        'correct_positions': correct_positions,
        'questions': questions,
        'answer_choices': answer_choices,
    }

    if verbose:
        print(f"  {len(correct_projections)} correct, {len(incorrect_projections)} incorrect projections", flush=True)
        summarize_projections(proj_result, offsets)

    return direction, proj_result


def extract_and_save_direction_streaming(
    model_name: str,
    data_path: str,
    output_dir: str,
    offsets: Optional[List[int]] = None,
    min_confidence: float = 3.0,
    skip_negated: bool = False,
    suffix: Optional[str] = None,
    existing_direction_path: Optional[str] = None,
    adaptive_offsets: bool = False,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    cache_dir: Optional[str] = None,
) -> None:
    """Memory-efficient direction extraction — drop-in replacement for extract_and_save_direction_hf."""

    if offsets is None:
        offsets = [-1, 0, 1, 2]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {data_path}...", flush=True)
    with open(data_path, 'rb') as f:
        raw = pickle.load(f)

    if isinstance(raw, dict) and 'pairs' in raw:
        print("Detected patching pairs format — converting...", flush=True)
        data = _convert_patching_pairs(raw['pairs'])
    else:
        data = raw
    print(f"Loaded {len(data)} datapoints", flush=True)

    if skip_negated:
        data, n_negated = _filter_negated(data)
        print(f"Negation filter: removed {n_negated}, {len(data)} remaining", flush=True)

    # Load model
    print(f"Loading model {model_name} (HF backend)...", flush=True)
    bundle = load_model_hf(
        model_name,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        cache_dir=cache_dir,
    )

    # Filtering
    n_loaded = len(data)
    filter_meta = {
        'n_loaded': n_loaded,
        'skip_negated': skip_negated,
        'min_confidence': min_confidence,
        'backend': 'huggingface_streaming',
    }

    if min_confidence > 0:
        from src.direction_finder_hf import add_confidence_and_filter_hf
        n_before = len(data)
        data, n_skipped = add_confidence_and_filter_hf(bundle, data, min_confidence)
        print(f"Confidence filter: {len(data)} / {n_before} kept", flush=True)
    else:
        print("Skipping confidence filter (min_confidence=0)", flush=True)

    if len(data) == 0:
        print("WARNING: All datapoints filtered out!", flush=True)
        return

    # Load existing direction if provided
    existing_direction = None
    if existing_direction_path:
        saved = torch.load(existing_direction_path, map_location='cpu')
        existing_direction = saved['directions']
        print(f"Loaded existing direction: {existing_direction.shape}", flush=True)

    # Run streaming extraction
    direction, proj_result = extract_direction_and_projections_streaming(
        bundle, data, offsets,
        adaptive_offsets=adaptive_offsets,
        existing_direction=existing_direction,
    )

    sfx = f'_{suffix}' if suffix else ''

    # Save direction
    if existing_direction_path is None:
        metadata = {
            'model_name': model_name,
            'n_datapoints': len(data),
            'filter_meta': filter_meta,
            'backend': 'huggingface_streaming',
        }
        direction_path = output_dir / f'direction{sfx}.pt'
        save_directions(direction, direction_path, offsets, proj_result['layers'], metadata)

    # Save projections
    proj_path = output_dir / f'projections{sfx}.pkl'
    with open(proj_path, 'wb') as f:
        pickle.dump(proj_result, f)
    print(f"Saved projections to {proj_path}", flush=True)

    # Plot
    plot_path = output_dir / f'projection_separation{sfx}.png'
    plot_projection_separation(proj_result, offsets, plot_path)

    # Summary
    summary = {
        'model_name': model_name,
        'n_layers': bundle.n_layers,
        'd_model': bundle.d_model,
        'backend': 'huggingface_streaming',
        'n_datapoints': len(data),
        'offsets': offsets,
        'streaming': True,
    }
    summary_path = output_dir / f'summary{sfx}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    del bundle
    gc.collect()
    torch.cuda.empty_cache()

    print("Done!", flush=True)
