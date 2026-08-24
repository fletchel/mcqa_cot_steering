"""HuggingFace version of data_generator.py: confidence filtering with HF models.

The generate stage (CPU-only, no model needed) is identical to data_generator.py.
Only the confidence filter stage needs a model, and this version uses HF directly.

Imports generate_pairs from data_generator.py (unchanged) and provides
generate_and_filter_pairs_hf() which replaces the TransformerLens confidence filter.
"""

import gc
from typing import Optional, Dict

import torch

from src.hf_utils import ModelBundle, run_forward_hf, get_letter_token_ids_hf

# Reuse CPU-only functions from existing module
from src.data_generator import (
    generate_pairs,
    _load_dataset_items,
    _extract_item,
    _format_question_no_cot,
    _apply_ordering,
    DATASET_INSTRUCTIONS,
)


def generate_and_filter_pairs_hf(
    bundle: ModelBundle,
    model_name: str,
    num_datapoints: int = 500,
    seed: int = 42,
    min_choice_tokens: int = 3,
    max_choice_tokens: Optional[int] = None,
    min_confidence: float = 3.0,
    require_equal_tokens: bool = True,
    skip_negated: bool = False,
    dataset: str = 'mmlu',
    n_choices: Optional[int] = None,
) -> Dict:
    """Generate shuffled MCQ items with all filtering in a single pass (HF version).

    Identical logic to data_generator.generate_and_filter_pairs but uses
    HuggingFace model directly instead of TransformerLens.
    """
    import random
    random.seed(seed)
    tokenizer = bundle.tokenizer

    ds, all_indices = _load_dataset_items(dataset, seed)

    letter_ids = None  # Initialized on first valid item

    pairs = []
    n_checked = 0
    n_skipped_punctuation = 0
    n_skipped_min_tokens = 0
    n_skipped_max_tokens = 0
    n_skipped_tokens = 0
    n_skipped_delimiters = 0
    n_skipped_confidence = 0
    n_skipped_negated = 0

    _NEGATION_WORDS = {'not', 'except', 'never', 'neither', 'nor', 'without', 'cannot'}

    from src.utils import get_prompt_answer_positions

    for idx in all_indices:
        if len(pairs) >= num_datapoints:
            break

        n_checked += 1
        item = ds[idx]
        question, choices, correct_idx, subject = _extract_item(dataset, item)

        # Skip items with wrong number of choices
        if n_choices is not None and len(choices) != n_choices:
            continue

        # Initialize or expand letter_ids to cover all choices
        if letter_ids is None or len(choices) > len(letter_ids):
            letter_ids = get_letter_token_ids_hf(tokenizer, bundle.device, n_choices=len(choices))

        if skip_negated:
            q_lower = question.lower()
            if any(w in q_lower.split() for w in _NEGATION_WORDS):
                n_skipped_negated += 1
                continue

        choices = [c.rstrip('.') for c in choices]

        token_lengths = [
            len(tokenizer.encode(" " + c, add_special_tokens=False)) for c in choices
        ]
        if any(tl < min_choice_tokens for tl in token_lengths):
            n_skipped_min_tokens += 1
            continue
        if max_choice_tokens is not None and any(tl > max_choice_tokens for tl in token_lengths):
            n_skipped_max_tokens += 1
            continue
        if require_equal_tokens and len(set(token_lengths)) != 1:
            n_skipped_tokens += 1
            continue

        n_ch = len(choices)
        order = list(range(n_ch))
        random.shuffle(order)
        correct_pos = order.index(correct_idx)
        answers = _apply_ordering(choices, order)
        gen = _format_question_no_cot(tokenizer, question, answers, dataset=dataset)

        pos = get_prompt_answer_positions(tokenizer, gen, answers)
        if not pos['success']:
            n_skipped_delimiters += 1
            continue

        pair_data = {
            'question': question,
            'subject': subject,
            'answer_texts': choices,
            'correct_idx_original': correct_idx,
            'gen': gen,
            'order': order,
            'correct_pos': correct_pos,
            'delimiter_positions': pos['positions'],
        }

        # GPU filter: confidence check using HF model
        input_ids = torch.tensor(
            tokenizer.encode(gen, add_special_tokens=False),
            device=bundle.device,
        ).unsqueeze(0)

        logits = run_forward_hf(bundle, input_ids)
        final_logits = logits[0, -1, letter_ids[:n_ch]]

        correct_logit = final_logits[correct_pos]
        other_logits = torch.cat([
            final_logits[:correct_pos],
            final_logits[correct_pos + 1:]
        ])
        confidence = (correct_logit - other_logits.max()).item()
        pair_data['confidence'] = confidence

        if confidence < min_confidence:
            n_skipped_confidence += 1
            continue

        pairs.append(pair_data)

        if len(pairs) % 10 == 0:
            print(f"  Collected {len(pairs)}/{num_datapoints} items ({n_checked} checked)",
                  flush=True)
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\nGenerated {len(pairs)} fully-filtered items from {n_checked} questions", flush=True)
    if skip_negated:
        print(f"  Skipped (negated question): {n_skipped_negated}", flush=True)
    print(f"  Skipped (comma in choice): {n_skipped_punctuation}", flush=True)
    print(f"  Skipped (min tokens < {min_choice_tokens}): {n_skipped_min_tokens}", flush=True)
    if max_choice_tokens is not None:
        print(f"  Skipped (max tokens > {max_choice_tokens}): {n_skipped_max_tokens}", flush=True)
    print(f"  Skipped (unequal token lengths): {n_skipped_tokens}", flush=True)
    print(f"  Skipped (delimiter parsing): {n_skipped_delimiters}", flush=True)
    print(f"  Skipped (confidence < {min_confidence}): {n_skipped_confidence}", flush=True)

    return {
        'pairs': pairs,
        'metadata': {
            'model_name': model_name,
            'dataset': dataset,
            'backend': 'huggingface',
            'num_checked': n_checked,
            'num_filtered': len(pairs),
            'seed': seed,
            'min_choice_tokens': min_choice_tokens,
            'max_choice_tokens': max_choice_tokens,
            'min_confidence': min_confidence,
            'n_skipped_punctuation': n_skipped_punctuation,
            'n_skipped_min_tokens': n_skipped_min_tokens,
            'n_skipped_tokens': n_skipped_tokens,
            'n_skipped_delimiters': n_skipped_delimiters,
            'n_skipped_confidence': n_skipped_confidence,
        },
    }
