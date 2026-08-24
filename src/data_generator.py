"""
Data generation and filtering for activation patching experiments.

Generates MCQ prompts with shuffled answer orderings and applies filtering
(token length, confidence). Each item is a single prompt with metadata.

Supported datasets:
  - mmlu: MMLU multiple-choice questions (cais/mmlu)
  - hellaswag: HellaSwag sentence completion (Rowan/hellaswag)

Two phases:
  1. generate_pairs — CPU only: generate items with shuffled orderings,
     apply token length filters
  2. generate_and_filter_pairs — GPU: generate + confidence filter in one pass
"""

import gc
import pickle
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


DATASET_INSTRUCTIONS = {
    'mmlu': "Answer the following question.",
    'hellaswag': "Complete the following sentence.",
    'truthfulqa': "Answer the following question.",
}


def _load_dataset_items(dataset: str, seed: int):
    """Load dataset and return (dataset_obj, shuffled_indices)."""
    random.seed(seed)
    if dataset == 'mmlu':
        print("Loading MMLU dataset...")
        ds = load_dataset('cais/mmlu', 'all')['test']
    elif dataset == 'hellaswag':
        print("Loading HellaSwag dataset...")
        ds = load_dataset('Rowan/hellaswag')['validation']
    elif dataset == 'truthfulqa':
        print("Loading TruthfulQA dataset...")
        ds = load_dataset('truthful_qa', 'multiple_choice')['validation']
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    all_indices = list(range(len(ds)))
    random.shuffle(all_indices)
    return ds, all_indices


def _extract_item(dataset: str, item: dict) -> Tuple[str, List[str], int, str]:
    """Extract (question, choices, correct_idx, subject) from a dataset item."""
    if dataset == 'mmlu':
        return (
            item['question'],
            list(item['choices']),
            item['answer'],
            item.get('subject', ''),
        )
    elif dataset == 'hellaswag':
        return (
            item['ctx'],
            list(item['endings']),
            int(item['label']),
            item.get('activity_label', ''),
        )
    elif dataset == 'truthfulqa':
        choices = item['mc1_targets']['choices']
        labels = item['mc1_targets']['labels']
        correct_idx = labels.index(1)
        return (
            item['question'],
            choices,
            correct_idx,
            '',
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def _format_question_no_cot(
    tokenizer,
    question: str,
    answers: List[str],
    dataset: str = 'mmlu',
) -> str:
    """Format an MCQ prompt using the model's chat template with no CoT hint.

    User turn: instruction + question + comma-delimited choices
    Assistant prefill: 'The final answer is ('
    """
    import string
    fmt = [f'({c})' for c in string.ascii_lowercase[:len(answers)]]
    choices_str = ', '.join(f'{fmt[i]} {answers[i]}' for i in range(len(answers)))

    instruction = DATASET_INSTRUCTIONS.get(dataset, DATASET_INSTRUCTIONS['mmlu'])
    prompt_text = (
        f"{instruction}\n"
        f"Question: {question}\n"
        f"Choices: {choices_str}\n"
    )

    messages = [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": "The final answer is ("},
    ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, continue_final_message=True
    )


def _apply_ordering(choices: List[str], ordering: List[int]) -> List[str]:
    """Reorder choices according to the given permutation.

    ordering[i] = j means position i gets the choice originally at index j.
    """
    return [choices[ordering[i]] for i in range(len(ordering))]


def generate_pairs(
    model_name: str,
    num_datapoints: int = 500,
    seed: int = 42,
    min_choice_tokens: int = 3,
    max_choice_tokens: Optional[int] = None,
    require_equal_tokens: bool = True,
    skip_negated: bool = False,
    dataset: str = 'mmlu',
    n_choices: Optional[int] = None,
) -> Dict:
    """Generate shuffled MCQ items with filtering (CPU only).

    For each question, shuffles the answer ordering and applies filters.

    Args:
        model_name: HuggingFace model ID (for tokenizer + chat template)
        num_datapoints: Number of questions to generate
        seed: Random seed
        min_choice_tokens: Minimum token count per answer choice
        dataset: Dataset to use ('mmlu' or 'hellaswag')

    Returns:
        Dict with 'pairs' list and 'metadata'
    """
    random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ds, all_indices = _load_dataset_items(dataset, seed)

    pairs = []
    n_checked = 0
    n_skipped_tokens = 0
    n_skipped_min_tokens = 0
    n_skipped_max_tokens = 0
    n_skipped_punctuation = 0
    n_skipped_negated = 0

    _NEGATION_WORDS = {'not', 'except', 'never', 'neither', 'nor', 'without', 'cannot'}

    from src.utils import get_prompt_answer_positions

    for idx in all_indices:
        if len(pairs) >= num_datapoints:
            break

        n_checked += 1
        item = ds[idx]
        question, choices, correct_idx, subject = _extract_item(dataset, item)

        # Skip items with wrong number of choices (for datasets with variable n_choices)
        if n_choices is not None and len(choices) != n_choices:
            continue

        # Skip negated questions (MMLU-specific but harmless for others)
        if skip_negated:
            q_lower = question.lower()
            if any(w in q_lower.split() for w in _NEGATION_WORDS):
                n_skipped_negated += 1
                continue

        # Strip trailing periods from choices
        choices = [c.rstrip('.') for c in choices]

        # Tokenize each choice (with leading space to match in-prompt tokenization)
        token_lengths = [
            len(tokenizer.encode(" " + c, add_special_tokens=False)) for c in choices
        ]

        # Min choice tokens filter
        if any(tl < min_choice_tokens for tl in token_lengths):
            n_skipped_min_tokens += 1
            continue

        # Max choice tokens filter
        if max_choice_tokens is not None and any(tl > max_choice_tokens for tl in token_lengths):
            n_skipped_max_tokens += 1
            continue

        # Equal token length filter
        if require_equal_tokens and len(set(token_lengths)) != 1:
            n_skipped_tokens += 1
            continue

        # Generate random ordering
        n_choices = len(choices)
        order = list(range(n_choices))
        random.shuffle(order)

        # Find correct answer position
        # order[i] = j means position i has the original choice j
        correct_pos = order.index(correct_idx)

        # Build reordered answer list
        answers = _apply_ordering(choices, order)

        # Format prompt
        gen = _format_question_no_cot(tokenizer, question, answers, dataset=dataset)

        # Find delimiter positions
        pos = get_prompt_answer_positions(tokenizer, gen, answers)

        if not pos['success']:
            continue

        pairs.append({
            'question': question,
            'subject': subject,
            'answer_texts': choices,
            'correct_idx_original': correct_idx,
            'gen': gen,
            'order': order,
            'correct_pos': correct_pos,
            'delimiter_positions': pos['positions'],
        })

    print(f"Generated {len(pairs)} items from {n_checked} questions checked")
    if skip_negated:
        print(f"  Skipped (negated question): {n_skipped_negated}")
    print(f"  Skipped (comma in choice): {n_skipped_punctuation}")
    print(f"  Skipped (min tokens < {min_choice_tokens}): {n_skipped_min_tokens}")
    if max_choice_tokens is not None:
        print(f"  Skipped (max tokens > {max_choice_tokens}): {n_skipped_max_tokens}")
    print(f"  Skipped (unequal token lengths): {n_skipped_tokens}")

    return {
        'pairs': pairs,
        'metadata': {
            'model_name': model_name,
            'dataset': dataset,
            'num_raw': n_checked,
            'num_generated': len(pairs),
            'seed': seed,
            'min_choice_tokens': min_choice_tokens,
            'max_choice_tokens': max_choice_tokens,
            'n_skipped_punctuation': n_skipped_punctuation,
            'n_skipped_min_tokens': n_skipped_min_tokens,
            'n_skipped_tokens': n_skipped_tokens,
        },
    }


def generate_and_filter_pairs(
    model,
    model_name: str,
    num_datapoints: int = 500,
    seed: int = 42,
    min_choice_tokens: int = 3,
    max_choice_tokens: Optional[int] = None,
    min_confidence: float = 3.0,
    require_equal_tokens: bool = True,
    skip_negated: bool = False,
    batch_size: int = 4,  # Currently unused, reserved for future batching
    dataset: str = 'mmlu',
    n_choices: Optional[int] = None,
) -> Dict:
    """Generate shuffled MCQ items with all filtering in a single pass.

    Iterates through dataset questions, applies CPU filters (punctuation, min tokens,
    equal token lengths) and GPU confidence filtering together. Stops when
    num_datapoints fully-filtered items are collected or data is exhausted.

    Args:
        model: TransformerLens HookedTransformer (for inference + tokenizer)
        model_name: HuggingFace model ID (for metadata)
        num_datapoints: Target number of fully-filtered items
        seed: Random seed
        min_choice_tokens: Minimum token count per answer choice
        min_confidence: Confidence threshold
        dataset: Dataset to use ('mmlu' or 'hellaswag')

    Returns:
        Dict with 'pairs' list and 'metadata'
    """
    random.seed(seed)
    tokenizer = model.tokenizer

    ds, all_indices = _load_dataset_items(dataset, seed)

    # letter_ids determined per-item based on n_choices (set after first item)
    letter_ids_tensor = None

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

        # Skip negated questions
        if skip_negated:
            q_lower = question.lower()
            if any(w in q_lower.split() for w in _NEGATION_WORDS):
                n_skipped_negated += 1
                continue

        # Strip trailing periods from choices
        choices = [c.rstrip('.') for c in choices]

        # CPU filter: min token count (leading space matches in-prompt tokenization)
        token_lengths = [
            len(tokenizer.encode(" " + c, add_special_tokens=False)) for c in choices
        ]
        if any(tl < min_choice_tokens for tl in token_lengths):
            n_skipped_min_tokens += 1
            continue

        # CPU filter: max token count
        if max_choice_tokens is not None and any(tl > max_choice_tokens for tl in token_lengths):
            n_skipped_max_tokens += 1
            continue

        # CPU filter: equal token lengths
        if require_equal_tokens and len(set(token_lengths)) != 1:
            n_skipped_tokens += 1
            continue

        # Generate random ordering
        n_choices = len(choices)
        order = list(range(n_choices))
        random.shuffle(order)
        correct_pos = order.index(correct_idx)
        answers = _apply_ordering(choices, order)
        gen = _format_question_no_cot(tokenizer, question, answers, dataset=dataset)

        # Initialize letter_ids on first item
        if letter_ids_tensor is None:
            import string
            letter_ids = tokenizer.convert_tokens_to_ids(
                list(string.ascii_lowercase[:n_choices]))
            letter_ids_tensor = torch.tensor(letter_ids, device=model.cfg.device)

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

        # GPU filter: confidence check
        with torch.no_grad():
            logits = model(gen, return_type='logits')
            final_logits = logits[0, -1, letter_ids_tensor]

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
            print(f"  Collected {len(pairs)}/{num_datapoints} items ({n_checked} checked)")
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\nGenerated {len(pairs)} fully-filtered items from {n_checked} questions")
    if skip_negated:
        print(f"  Skipped (negated question): {n_skipped_negated}")
    print(f"  Skipped (comma in choice): {n_skipped_punctuation}")
    print(f"  Skipped (min tokens < {min_choice_tokens}): {n_skipped_min_tokens}")
    if max_choice_tokens is not None:
        print(f"  Skipped (max tokens > {max_choice_tokens}): {n_skipped_max_tokens}")
    print(f"  Skipped (unequal token lengths): {n_skipped_tokens}")
    print(f"  Skipped (delimiter parsing): {n_skipped_delimiters}")
    print(f"  Skipped (confidence < {min_confidence}): {n_skipped_confidence}")

    return {
        'pairs': pairs,
        'metadata': {
            'model_name': model_name,
            'dataset': dataset,
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


