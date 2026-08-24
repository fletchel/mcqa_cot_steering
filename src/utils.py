"""Shared utilities: answer-delimiter position finding and direction I/O.

Extracted verbatim from the previously vendored mech_interp package
(eap_utils.get_prompt_answer_positions, utils.direction_utils.save_directions /
summarize_projections) — the only pieces of it the pipeline uses.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch


def get_prompt_answer_positions(
    tokenizer,
    generation: str,
    choices: List[str],
    verify_delimiters: bool = True
) -> Dict:
    """
    Find the positions of tokens immediately AFTER each of the 4 MCQ answer choices.

    Uses raw string search + character-to-token mapping to handle edge cases where
    tokenization differs based on context (e.g., when choice text ends with a period).

    This function:
    1. Searches in raw string (not tokenized)
    2. Finds the LAST (final) occurrence of each choice
    3. Maps character positions to token positions using offset mapping

    Args:
        tokenizer: The model's tokenizer
        generation: The full generation text containing the MCQ prompt
        choices: List of 4 answer choice texts (from shuffled_answers)
        verify_delimiters: If True, check that positions contain expected delimiters

    Returns:
        Dict with:
            - 'positions': List of 4 token positions (or -1 if not found)
            - 'tokens_at_positions': List of tokens at those positions
            - 'is_delimiter': List of bools indicating if token is comma/period
            - 'success': Bool indicating if all 4 positions were found
            - 'tokenized_length': Length of tokenized generation
    """
    # Get tokenized output with offset mapping
    # offset_mapping gives (start_char, end_char) for each token
    encoding = tokenizer(
        generation,
        return_offsets_mapping=True,
        add_special_tokens=False
    )
    token_ids = encoding['input_ids']
    offset_mapping = encoding['offset_mapping']

    # Get the actual tokens for output
    tokenized = tokenizer.convert_ids_to_tokens(token_ids)

    positions = []
    tokens_at_pos = []
    is_delimiter = []

    for choice_text in choices:
        # Find the LAST occurrence of the choice in the raw string
        # We search for the choice with a leading space (as it appears after the letter label)
        search_text = ' ' + choice_text
        last_idx = generation.rfind(search_text)

        if last_idx == -1:
            # Try without leading space
            search_text = choice_text
            last_idx = generation.rfind(search_text)

        if last_idx == -1:
            # Choice not found in raw string
            positions.append(-1)
            tokens_at_pos.append(None)
            is_delimiter.append(False)
            continue

        # Calculate the character position where the choice TEXT ends
        # (not including the leading space we added for search)
        if search_text.startswith(' '):
            char_end = last_idx + len(search_text)
        else:
            char_end = last_idx + len(choice_text)

        # Map character position to token position
        # Find the token that contains or starts at char_end
        # The delimiter token is the one whose start_char >= char_end
        delimiter_token_idx = -1
        for tok_idx, (start_char, end_char) in enumerate(offset_mapping):
            if start_char >= char_end:
                delimiter_token_idx = tok_idx
                break

        if delimiter_token_idx == -1:
            # Could not map to a token (choice is at the very end)
            positions.append(-1)
            tokens_at_pos.append(None)
            is_delimiter.append(False)
            continue

        positions.append(delimiter_token_idx)
        token = tokenized[delimiter_token_idx]
        tokens_at_pos.append(token)

        if verify_delimiters:
            # Check if it's a typical delimiter (comma, period, or variants)
            is_delim = any(d in token.lower() for d in [',', '.', ';', ')'])
            is_delimiter.append(is_delim)
        else:
            is_delimiter.append(True)

    success = all(p != -1 for p in positions)

    return {
        'positions': positions,
        'tokens_at_positions': tokens_at_pos,
        'is_delimiter': is_delimiter,
        'success': success,
        'tokenized_length': len(tokenized),
    }


def save_directions(
    directions: torch.Tensor,
    output_path: Union[str, Path],
    offsets: List[int],
    layers: List[int],
    metadata: Optional[Dict] = None
):
    """
    Save direction tensor with metadata.

    Args:
        directions: Tensor of shape (n_offsets, n_layers, d_model)
        output_path: Path to save .pt file
        offsets: List of position offsets
        layers: List of layer indices
        metadata: Optional additional metadata
    """
    save_dict = {
        'directions': directions,
        'offsets': offsets,
        'layers': layers,
        'shape': list(directions.shape),
        'metadata': metadata or {}
    }

    torch.save(save_dict, output_path)
    print(f"Saved directions to {output_path}")
    print(f"  Shape: {directions.shape} (offsets x layers x d_model)")


def summarize_projections(proj_result: Dict, offsets: Optional[List[int]] = None):
    """Print summary of direction projections."""
    offsets = offsets or proj_result['offsets']

    print("Direction Projection Analysis")
    print("=" * 50)
    print(f"Correct samples: {proj_result['correct_projections'].shape[0]}")
    print(f"Incorrect samples: {proj_result['incorrect_projections'].shape[0]}")
    print()

    for i, offset in enumerate(offsets):
        correct_mean = proj_result['correct_mean'][i].mean().item()
        incorrect_mean = proj_result['incorrect_mean'][i].mean().item()
        separation = proj_result['separation'][i].mean().item()

        print(f"Offset {offset:+d}:")
        print(f"  Correct mean proj:   {correct_mean:+.4f}")
        print(f"  Incorrect mean proj: {incorrect_mean:+.4f}")
        print(f"  Separation:          {separation:+.4f}")
        print()
