"""
Core pipeline for finding correctness directions in arbitrary instruct models.

Two stages:
  1. generate_model_data — CPU only: generates + filters localization data
  2. extract_and_save_direction — GPU: extracts residual streams, computes direction, saves results
"""

import gc
import json
import pickle
import re
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def model_slug(model_name: str) -> str:
    """Convert 'Qwen/Qwen2.5-7B-Instruct' -> 'qwen2.5-7b-instruct'."""
    return model_name.split('/')[-1].lower()


def strip_trailing_special_tokens(gen: str, tokenizer) -> str:
    """Strip any trailing special tokens from a generated string.

    Handles: <|im_end|>, <|eot_id|>, </s>, <|end|>, <|endoftext|>, etc.
    Applied as a generic post-processing step for broad model support.
    """
    special_tokens = set(tokenizer.all_special_tokens)
    changed = True
    while changed:
        changed = False
        for token in special_tokens:
            if gen.endswith(token):
                gen = gen[:-len(token)]
                changed = True
        # Also strip trailing whitespace/newlines left after token removal
        stripped = gen.rstrip()
        if stripped != gen:
            gen = stripped
            changed = True
    return gen


def get_letter_token_ids(model) -> list:
    """Get token IDs for answer letters ['a', 'b', 'c', 'd']."""
    return model.tokenizer.convert_tokens_to_ids(['a', 'b', 'c', 'd'])


def strip_cot_hint(text: str) -> str:
    """Remove CoT hint like 'I think the answer is (a). ' from text."""
    return re.sub(r"I think the answer is \([a-d]\)\.\s*", "", text)


def add_confidence_and_filter(model, data: list, min_confidence: float = 3.0) -> tuple:
    """Compute confidence for each clean sample and filter by threshold.

    Confidence = logit(correct_answer) - logit(next_best_answer).
    Computes letter-logit confidence for each datapoint.

    Returns:
        (filtered_data, n_skipped)
    """
    letter_ids = get_letter_token_ids(model)
    filtered = []
    skipped = 0

    for i, dp in enumerate(data):
        gen = strip_cot_hint(dp['pair'][0]['gen'])
        prompt_idx = dp['pair'][0]['prompt_pos']

        with torch.no_grad():
            logits = model(gen, return_type='logits')[0, -1, letter_ids]

        other_logits = torch.cat([logits[:prompt_idx], logits[prompt_idx + 1:]])
        confidence = logits[prompt_idx] - other_logits.max()

        if min_confidence > 0 and confidence.item() < min_confidence:
            skipped += 1
            continue

        dp['pair'][0]['confidence'] = confidence
        dp['pair'][0]['letter_logits'] = logits
        filtered.append(dp)

        if (i + 1) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    return filtered, skipped


def _convert_patching_pairs(pairs: list) -> list:
    """Convert patching pairs format to direction extraction format.

    The direction extractor expects items with:
        item['pair'][0]['gen'], item['pair'][0]['prompt_pos'],
        item['pair'][0]['shuffled_answers']

    Patching pairs have:
        pair['gen'], pair['correct_pos'], pair['order'], pair['answer_texts']
    """
    converted = []
    for pair in pairs:
        answer_texts = pair['answer_texts']
        # Handle both flat format (pair['order']) and nested (pair['elem1']['order'])
        if 'order' in pair:
            elem = pair
        elif 'elem1' in pair:
            elem = pair['elem1']
        else:
            raise KeyError(f"Expected 'order' or 'elem1' in pair, got keys: {list(pair.keys())}")
        n_ch = len(elem['order'])
        answers = [answer_texts[elem['order'][i]] for i in range(n_ch)]

        converted.append({
            'pair': [
                {
                    'gen': elem['gen'],
                    'prompt_pos': elem['correct_pos'],
                    'shuffled_answers': answers,
                },
            ],
            'question': pair.get('question', ''),
        })
    return converted


_NEGATION_WORDS = {'not', 'except', 'never', 'neither', 'nor', 'without', 'cannot'}


def _filter_negated(data: list) -> tuple[list, int]:
    """Remove datapoints whose question contains negation words.

    Checks 'original_question' (if present) or falls back to extracting
    the question from pair[0]['gen'].
    """
    kept = []
    n_removed = 0
    for item in data:
        question = item.get('original_question', '')
        if not question and 'pair' in item and item['pair']:
            # Fall back: extract from generated prompt
            gen = item['pair'][0].get('gen', '')
            question = gen
        words = set(question.lower().split())
        if words & _NEGATION_WORDS:
            n_removed += 1
        else:
            kept.append(item)
    return kept, n_removed


def plot_projection_separation(
    proj_result: Dict,
    offsets: List[int],
    output_path: str,
) -> None:
    """Plot projection separation (correct - incorrect) by layer, one line per offset."""
    separation = proj_result['separation']  # (n_offsets, n_layers)
    layers = proj_result['layers']

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, offset in enumerate(offsets):
        if torch.is_tensor(separation):
            sep_vals = separation[i].cpu().numpy()
        else:
            sep_vals = np.array(separation[i])
        ax.plot(layers, sep_vals, label=f'offset={offset}', marker='o', markersize=3)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Projection Separation (correct - incorrect)')
    ax.set_title('Correctness Direction: Projection Separation by Layer')
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _build_summary(
    model_name: str,
    model,
    data: list,
    extraction: Dict,
    proj_result: Dict,
    offsets: List[int],
    filter_meta: Optional[Dict] = None,
) -> Dict:
    """Build summary dictionary for JSON output."""
    layers = proj_result['layers']

    # Per-offset separation metrics
    separation_by_offset = {}
    for i, offset in enumerate(offsets):
        if torch.is_tensor(proj_result['separation']):
            sep_vals = proj_result['separation'][i].cpu().tolist()
        else:
            sep_vals = [float(v) for v in proj_result['separation'][i]]

        max_idx = int(np.argmax(sep_vals))
        separation_by_offset[str(offset)] = {
            'layers': [int(l) for l in layers],
            'separation': [float(v) for v in sep_vals],
            'max_separation': float(sep_vals[max_idx]),
            'max_layer': int(layers[max_idx]),
        }

    # Per-offset mean projections
    correct_means = {}
    incorrect_means = {}
    for i, offset in enumerate(offsets):
        if torch.is_tensor(proj_result['correct_mean']):
            correct_means[str(offset)] = [float(v) for v in proj_result['correct_mean'][i].cpu().tolist()]
            incorrect_means[str(offset)] = [float(v) for v in proj_result['incorrect_mean'][i].cpu().tolist()]
        else:
            correct_means[str(offset)] = [float(v) for v in proj_result['correct_mean'][i]]
            incorrect_means[str(offset)] = [float(v) for v in proj_result['incorrect_mean'][i]]

    return {
        'model_name': model_name,
        'n_layers': model.cfg.n_layers,
        'd_model': model.cfg.d_model,
        'filtering': filter_meta,
        'n_datapoints': len(data),
        'n_correct': int(extraction['correct'].shape[0]),
        'n_incorrect': int(extraction['incorrect'].shape[0]),
        'offsets': offsets,
        'separation_by_offset': separation_by_offset,
        'correct_mean_projections': correct_means,
        'incorrect_mean_projections': incorrect_means,
    }
