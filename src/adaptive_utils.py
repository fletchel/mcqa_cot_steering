"""Adaptive offset utilities for variable-length answer tokens.

When answers have different token lengths (e.g., 3-5 tokens), adaptive offsets
clip the patching/steering offsets per answer position to cover only that answer's
actual tokens. A 3-token answer uses [-3, -2, -1, 0]; a 5-token answer uses
[-5, -4, -3, -2, -1, 0].

delimiter_positions[i] points to the first token AFTER answer i's text (typically
a comma). So offset -1 = last answer token, offset -2 = second-to-last, etc.
"""

import os
from typing import Dict, List


def get_answer_token_length(tokenizer, answer_text: str) -> int:
    """Get token length of an answer text with leading space (matches in-prompt tokenization)."""
    return len(tokenizer.encode(" " + answer_text, add_special_tokens=False))


def get_answer_token_length_at_position(tokenizer, pair: dict, letter_pos: int) -> int:
    """Get the token length of the answer at a specific letter position.

    pair['order'][letter_pos] gives the original answer index at that position.
    """
    order = pair['order']
    answer_text = pair['answer_texts'][order[letter_pos]]
    return get_answer_token_length(tokenizer, answer_text)


def clip_offsets_for_token_length(full_offsets: List[int], token_length: int) -> List[int]:
    """Return the subset of offsets covering token_length answer tokens + delimiter.

    For token_length=3: returns [-3, -2, -1, 0] ∩ full_offsets.
    For token_length=5: returns [-5, -4, -3, -2, -1, 0] ∩ full_offsets.
    """
    valid_range = set(range(-token_length, 1))  # {-N, ..., 0}
    return [o for o in full_offsets if o in valid_range]


def get_min_offsets(full_offsets: List[int], tokenizer, pairs: list) -> List[int]:
    """Get offsets clipped to the shortest answer token length across all pairs."""
    min_length = float('inf')
    for pair in pairs:
        n_ch = len(pair['order'])
        for pos in range(n_ch):
            tl = get_answer_token_length_at_position(tokenizer, pair, pos)
            min_length = min(min_length, tl)
    return clip_offsets_for_token_length(full_offsets, int(min_length))


def group_pairs_by_token_length(tokenizer, pairs: list) -> Dict[int, List[int]]:
    """Group pair indices by correct answer token length."""
    groups = {}
    for i, pair in enumerate(pairs):
        tl = get_answer_token_length_at_position(
            tokenizer, pair, pair['correct_pos']
        )
        groups.setdefault(tl, []).append(i)
    return groups


def verify_token_lengths(tokenizer, pairs: list):
    """Verify token lengths and return indices of pairs with mismatches.

    Checks that tokenizer.encode(" " + answer_text) produces the same tokens
    as the actual span in the prompt (measured from delimiter positions).

    Returns:
        set of pair indices where any answer position has a mismatch.
    """
    # Count distribution
    length_counts = {}
    for pair in pairs:
        n_ch = len(pair['order'])
        for pos in range(n_ch):
            tl = get_answer_token_length_at_position(tokenizer, pair, pos)
            length_counts[tl] = length_counts.get(tl, 0) + 1

    print("  Answer token length distribution:", flush=True)
    for tl in sorted(length_counts):
        print(f"    {tl} tokens: {length_counts[tl]} answers", flush=True)

    # Verify all pairs against in-context tokenization
    mismatch_indices = set()
    for pair_idx, pair in enumerate(pairs):
        token_ids = tokenizer.encode(pair['gen'], add_special_tokens=False)
        delim_positions = pair['delimiter_positions']

        for pos in range(len(pair['order'])):
            answer_text = pair['answer_texts'][pair['order'][pos]]
            expected_tl = get_answer_token_length(tokenizer, answer_text)

            # Decode tokens at offsets [-expected_tl, ..., -1] from delimiter
            delim = delim_positions[pos]
            actual_tokens = []
            for off in range(-expected_tl, 0):
                tok_pos = delim + off
                if 0 <= tok_pos < len(token_ids):
                    actual_tokens.append(tokenizer.decode([token_ids[tok_pos]]))

            actual_text = "".join(actual_tokens)
            expected_text = " " + answer_text

            if actual_text != expected_text:
                mismatch_indices.add(pair_idx)

    if not mismatch_indices:
        print(f"  Verified {len(pairs)} pairs: all match", flush=True)
    else:
        print(f"  Removing {len(mismatch_indices)} pairs with token length mismatches", flush=True)

    return mismatch_indices


def save_debug_file(tokenizer, model, pairs: list, offsets: List[int], output_path: str):
    """Save a debug file showing tokenized examples with patching/steering positions marked.

    For each pair, shows:
    - Question, correct/target answers with token decomposition
    - Each elem's delimiter positions and order mapping
    - For each answer position: decoded tokens around the delimiter,
      the clipped offsets, and the actual tokens at patched positions
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = []
    lines.append("Adaptive Offsets Debug File")
    lines.append(f"Full offsets: {offsets}")
    lines.append(f"Num pairs: {len(pairs)}")
    lines.append("")

    # Tokenization comparison for first pair — verify positions match
    if pairs:
        first_gen = pairs[0]['gen']
        raw_ids = tokenizer.encode(first_gen, add_special_tokens=False)
        model_ids = model.to_tokens(first_gen)[0].tolist()
        lines.append("TOKENIZATION CHECK (first pair):")
        lines.append(f"  tokenizer.encode(gen, add_special_tokens=False): {len(raw_ids)} tokens")
        lines.append(f"  model.to_tokens(gen):                            {len(model_ids)} tokens")
        match = (raw_ids == model_ids)
        lines.append(f"  Exact match: {match}")
        if not match:
            lines.append(f"  WARNING: tokenizations differ!")
        # Show tokens around first delimiter to verify alignment
        delim0 = pairs[0]['delimiter_positions'][0]
        lines.append(f"  First delimiter_position: {delim0}")
        lines.append(f"  Token at raw[{delim0}]:   {repr(tokenizer.decode([raw_ids[delim0]]))}")
        lines.append(f"  Token at model[{delim0}]: {repr(tokenizer.decode([model_ids[delim0]]))}")
        lines.append("")

    max_abs_offset = max(abs(o) for o in offsets) if offsets else 0

    for pair_idx, pair in enumerate(pairs):
        lines.append("=" * 70)
        lines.append(f"=== Pair {pair_idx} ===")
        q = pair['question']
        lines.append(f"Question: {q[:120]}{'...' if len(q) > 120 else ''}")

        # Show all answer texts with token counts
        for ai, at in enumerate(pair['answer_texts']):
            tl = get_answer_token_length(tokenizer, at)
            toks = tokenizer.encode(" " + at, add_special_tokens=False)
            decoded = [repr(tokenizer.decode([t])) for t in toks]
            marker = " <-- correct" if ai == pair['correct_idx_original'] else ""
            lines.append(f"  Answer {ai}: {repr(at)}  [{tl} tokens: {', '.join(decoded)}]{marker}")

        lines.append("")
        lines.append(f"  order: {pair['order']}  correct_pos={pair['correct_pos']}")
        lines.append(f"  delimiter_positions: {pair['delimiter_positions']}")

        # Tokenize the full prompt (without special tokens to match delimiter positions)
        token_ids = tokenizer.encode(pair['gen'], add_special_tokens=False)

        import string as _string
        for pos in range(len(pair['order'])):
            letter = _string.ascii_lowercase[pos]
            answer_text = pair['answer_texts'][pair['order'][pos]]
            tl = get_answer_token_length(tokenizer, answer_text)
            clipped = clip_offsets_for_token_length(offsets, tl)
            delim = pair['delimiter_positions'][pos]

            # Show tokens around this position
            ctx_start = max(0, delim - max_abs_offset - 2)
            ctx_end = min(len(token_ids), delim + 3)

            tok_strs = []
            for t in range(ctx_start, ctx_end):
                decoded = repr(tokenizer.decode([token_ids[t]]))
                marker = "*" if t == delim else ""
                tok_strs.append(f"[{t}]={decoded}{marker}")

            lines.append(f"    ({letter}) delim={delim}, answer={repr(answer_text)} "
                         f"[{tl} tok]:")
            lines.append(f"      Tokens: {' '.join(tok_strs)}")

            # Show clipped offsets and what tokens they hit
            positions = [delim + o for o in clipped]
            patched_tokens = []
            for p in positions:
                if 0 <= p < len(token_ids):
                    patched_tokens.append(repr(tokenizer.decode([token_ids[p]])))
                else:
                    patched_tokens.append("OOB")

            lines.append(f"      Clipped offsets: {clipped} -> positions {positions}")
            lines.append(f"      Tokens at positions: {patched_tokens}")

        lines.append("")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved adaptive debug file: {output_path}", flush=True)
