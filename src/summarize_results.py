"""Print and save a summary of logit steering and CoT steering results."""

import argparse
import os
import pickle
import re

import torch


def load_pickle(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def parse_answers(results):
    """Parse answers from CoT steering results, returning counts.

    'target_fp' counts false positives: the model outputs the target letter
    but follows it with the correct answer's text (letter swapped, reasoning intact).
    These are included in the 'target' count as well — target_fp is a subset.
    """
    letters = ['a', 'b', 'c', 'd']
    counts = {'correct': 0, 'target': 0, 'target_fp': 0, 'other': 0, 'unclear': 0}
    total = 0

    for r in results:
        if not r.get('success', False):
            continue
        total += 1
        response = r.get('response', '')
        correct_idx = r['correct_idx']
        target_idx = r['target_idx']
        answer_texts = r.get('answer_texts')

        answer = extract_answer(response, letters, answer_texts)
        if answer is None:
            counts['unclear'] += 1
        elif answer == correct_idx:
            counts['correct'] += 1
        elif answer == target_idx:
            counts['target'] += 1
            if is_false_positive(response, answer, correct_idx, target_idx, answer_texts):
                counts['target_fp'] += 1
        else:
            counts['other'] += 1

    return counts, total


def extract_answer(response, letters=None, answer_texts=None, require_think_close=False):
    """Extract the final answer letter from a CoT response.

    Returns the index (0-3) of the chosen letter, or None if unparseable.
    Key design: finds all letter mentions, then walks backwards from the end
    to return the last one in a conclusory context. This ensures that when
    the model states one answer and then changes its mind, we take the last one.

    If answer_texts is provided (list of answer choice strings), also tries to
    match quoted completion text in the response against the choices as a fallback.
    """
    if letters is None:
        letters = ['a', 'b', 'c', 'd']

    # Thinking models (QwQ, Qwen3.6, ...): only the text after the final </think>
    # is the answer; mentions inside the think block are deliberation, not the
    # final choice. Note the chat template can place the opening <think> in the
    # prompt, so a truncated response may contain neither tag — callers that know
    # the run is a thinking run pass require_think_close=True to count those as
    # unparseable instead of extracting from mid-reasoning text.
    # GPT-OSS / Harmony: special tokens are stripped from `response`, leaving
    # 'analysis{deliberation}assistantfinal{answer}'. Only final-channel text
    # holds the answer; analysis with no final channel = truncated mid-reasoning.
    if 'assistantfinal' in response:
        response = response.rsplit('assistantfinal', 1)[1]
    elif response.lstrip().startswith('analysis'):
        return None
    elif '</think>' in response:
        response = response.rsplit('</think>', 1)[1]
    elif '<think>' in response or require_think_close:
        return None

    resp_lower = response.strip().lower()
    if not resp_lower:
        return None

    # LaTeX-boxed final answer (\boxed{c} or \boxed{(c)}), used by Qwen3-32B
    # and other math-flavored models. Inherently conclusory: take the last one.
    boxed = re.findall(r'\\boxed\{\s*\(?([' + ''.join(letters) + r'])\)?\s*\}', resp_lower)
    if boxed:
        return letters.index(boxed[-1])

    # Immediate single letter (for short, non-CoT responses)
    if resp_lower[0] in letters:
        remaining = resp_lower[1:].lstrip()
        if not remaining or remaining[0] in '.):,\n ':
            return letters.index(resp_lower[0])

    # Collect all letter mentions with positions: (position, letter)
    letters_set = set(letters)
    letter_pattern = '[' + ''.join(letters) + ']'
    mentions = []
    for m in re.finditer(r'\((' + letter_pattern + r')\)', resp_lower):
        mentions.append((m.start(), m.group(1)))
    for m in re.finditer(r'\*\*\(?(' + letter_pattern + r')\)?\*\*', resp_lower):
        mentions.append((m.start(), m.group(1)))
    mentions.sort(key=lambda x: x[0])

    # Walk backwards: return the last mention with conclusory context
    conclusory_keywords = [
        'answer is', 'answer would be', 'choice is', 'choice would be',
        'therefore', 'thus,', 'hence,', 'final answer', 'so,',
        'correct choice', 'best choice', 'best answer', 'correct answer',
        'most appropriate', 'most logical', 'best completion',
        'completed sentence', 'best option',
    ]
    for pos, letter in reversed(mentions):
        context = resp_lower[max(0, pos - 100):pos]
        if any(kw in context for kw in conclusory_keywords):
            return letters.index(letter)

    # Try matching quoted answer text in the final portion of the response.
    # Handles cases where the model quotes the full completion text without
    # restating the letter (common in HellaSwag).
    # Uses the last ~6 words as a distinctive suffix, since the model
    # sometimes paraphrases the beginning.
    if answer_texts:
        last_500 = resp_lower[-500:]
        best_idx = None
        best_text_pos = -1
        for i, text in enumerate(answer_texts):
            text_lower = text.strip().lower()
            for snippet in [text_lower, ' '.join(text_lower.split()[-6:])]:
                if len(snippet) < 5:
                    continue
                pos = last_500.rfind(snippet)
                if pos > best_text_pos:
                    best_text_pos = pos
                    best_idx = i
        if best_idx is not None:
            return best_idx

    # Fallback: last (letter) mention in the final third of the response
    if mentions:
        cutoff = len(resp_lower) * 2 // 3
        for pos, letter in reversed(mentions):
            if pos >= cutoff:
                return letters.index(letter)
        # Last resort: last mention overall
        return letters.index(mentions[-1][1])

    return None


# Keep old name as alias for backwards compatibility
_extract_answer = extract_answer


def is_false_positive(response, parsed_idx, correct_idx, target_idx, answer_texts):
    """Check if a parsed "target" answer is a false positive.

    A false positive occurs when the model outputs the target letter but follows
    it with the correct answer's text, e.g. "(b) Dublin" where (b) is the target
    letter but Dublin is the correct answer text. This indicates the model still
    "knows" the correct answer but the letter got swapped.

    Returns True if the parsed answer matches the target letter but the text
    following it matches the correct answer rather than the target answer.
    Only meaningful when parsed_idx == target_idx.
    """
    if parsed_idx != target_idx or not answer_texts:
        return False
    if correct_idx >= len(answer_texts) or target_idx >= len(answer_texts):
        return False

    correct_text = answer_texts[correct_idx].strip().lower()
    target_text = answer_texts[target_idx].strip().lower()
    if len(correct_text) < 3 or len(target_text) < 3:
        return False

    resp_lower = response.strip().lower()
    import string as _string
    n_ch = max(correct_idx, target_idx) + 1
    if answer_texts:
        n_ch = max(n_ch, len(answer_texts))
    letters = list(_string.ascii_lowercase[:n_ch])
    target_letter = letters[target_idx]

    # Find the last occurrence of the target letter in (letter) or **letter** form
    last_pos = -1
    for m in re.finditer(r'\(' + target_letter + r'\)', resp_lower):
        last_pos = m.end()
    for m in re.finditer(r'\*\*\(?' + target_letter + r'\)?\*\*', resp_lower):
        last_pos = max(last_pos, m.end())

    if last_pos < 0:
        return False

    # Check the text after the final target letter mention
    after_text = resp_lower[last_pos:last_pos + 150]

    # Use the first few words of each answer as an identifier
    def get_signature(text, n_words=4):
        words = text.split()[:n_words]
        return ' '.join(words) if len(words) >= 2 else text[:20]

    correct_sig = get_signature(correct_text)
    target_sig = get_signature(target_text)

    has_correct = correct_sig in after_text
    has_target = target_sig in after_text

    return has_correct and not has_target


def _get_baseline_correct_indices(args):
    """Return set of batch_idx values where baseline got the answer correct, or None."""
    if not os.path.isdir(args.cot_dir):
        return None
    baseline_files = [
        f for f in os.listdir(args.cot_dir)
        if f.endswith('.pkl') and 'baseline' in f and 'intervened' in f
    ]
    if not baseline_files:
        return None
    data = load_pickle(os.path.join(args.cot_dir, baseline_files[0]))
    if not data or 'successful_results' not in data:
        return None

    letters = ['a', 'b', 'c', 'd']
    correct_indices = set()
    for r in data['successful_results']:
        if not r.get('success', False):
            continue
        answer = _extract_answer(r.get('response', ''), letters)
        if answer is not None and answer == r['correct_idx']:
            correct_indices.add(r.get('batch_idx'))
    return correct_indices if correct_indices else None


def format_table(rows, headers):
    """Format a list of row dicts into an aligned ASCII table."""
    col_widths = {h: len(h) for h in headers}
    str_rows = []
    for row in rows:
        str_row = {}
        for h in headers:
            val = row.get(h, '')
            s = str(val)
            str_row[h] = s
            col_widths[h] = max(col_widths[h], len(s))
        str_rows.append(str_row)

    sep = '+-' + '-+-'.join('-' * col_widths[h] for h in headers) + '-+'
    header_line = '| ' + ' | '.join(h.ljust(col_widths[h]) for h in headers) + ' |'

    lines = [sep, header_line, sep]
    for sr in str_rows:
        line = '| ' + ' | '.join(sr[h].ljust(col_widths[h]) for h in headers) + ' |'
        lines.append(line)
    lines.append(sep)
    return '\n'.join(lines)


def build_summary(args):
    """Build the full summary string with hyperparameters and results table."""
    lines = []

    # Hyperparameters section
    lines.append("=" * 70)
    lines.append("  PIPELINE SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    lines.append("HYPERPARAMETERS")
    lines.append("-" * 40)
    if args.model_name:
        lines.append(f"  Model:              {args.model_name}")
    if args.suffix:
        lines.append(f"  Run name:           {args.suffix}")
    if args.num_datapoints:
        lines.append(f"  Num datapoints:     {args.num_datapoints}")
    if args.min_confidence:
        lines.append(f"  Min confidence:     {args.min_confidence}")
    if args.min_choice_tokens:
        lines.append(f"  Min choice tokens:  {args.min_choice_tokens}")
    if args.max_choice_tokens:
        lines.append(f"  Max choice tokens:  {args.max_choice_tokens}")
    if args.seed:
        lines.append(f"  Seed:               {args.seed}")
    if args.test_fraction:
        lines.append(f"  Test fraction:      {args.test_fraction}")
    if args.direction_offsets:
        lines.append(f"  Direction offsets:   {args.direction_offsets}")
    if args.patch_mode:
        lines.append(f"  Patch mode:         {args.patch_mode}")
    if args.patch_offsets:
        lines.append(f"  Patch offsets:       {args.patch_offsets}")
    lines.append("")

    # Patching section
    if args.patching_dir and os.path.isdir(args.patching_dir):
        patching_files = sorted(
            f for f in os.listdir(args.patching_dir) if f.endswith('.pkl')
        )
        if patching_files:
            lines.append("PATCHING EXPERIMENTS")
            lines.append("-" * 40)

            descriptions = {
                'random_swap': 'Per-offset: patch one offset at a time from random other question',
                'random_swap_alloff': 'All-offsets: patch all offsets simultaneously from random other question',
                'random_swap_min_offsets': 'Per-offset (min-offsets): non-adaptive, offsets safe for all answers',
                'random_swap_alloff_min_offsets': 'All-offsets (min-offsets): non-adaptive, offsets safe for all answers',
                'swap': 'Per-offset: patch one offset at a time (legacy)',
                'swap_alloff': 'All-offsets: patch all offsets simultaneously (legacy)',
            }

            for pf in patching_files:
                data = load_pickle(os.path.join(args.patching_dir, pf))
                if data is None:
                    continue
                meta = data.get('metadata', {})
                n_pairs = meta.get('n_pairs', '?')
                n_skipped = meta.get('n_skipped', 0)
                offsets = meta.get('offsets', []) if not meta.get('all_offsets') else meta.get('offsets', [])
                adaptive = meta.get('adaptive_offsets', False)
                all_off = meta.get('all_offsets', False)

                # Derive experiment key from filename
                name = pf.replace('patching_results_', '').replace('.pkl', '')
                desc = descriptions.get(name, name)
                adaptive_str = ' [adaptive]' if adaptive else ''

                mean_metric = ''
                if 'mean_metrics' in data:
                    mm = data['mean_metrics']
                    mean_metric = f", mean metric = {torch.nanmean(mm).item():.4f}"

                lines.append(f"  {pf}")
                lines.append(f"    {desc}{adaptive_str}")
                offsets_list = data.get('offsets', meta.get('offsets', '?'))
                tok_len_counts = meta.get('token_length_counts', {})
                tok_len_str = ""
                if tok_len_counts:
                    parts = [f"{k}-tok: {v}" for k, v in sorted(tok_len_counts.items())]
                    tok_len_str = f" ({', '.join(parts)})"
                lines.append(f"    N = {n_pairs} pairs{tok_len_str} ({n_skipped} skipped), "
                             f"offsets = {offsets_list}{mean_metric}")
            lines.append("")

    # Results table
    rows = []
    sweep_params = []  # (tag, params_str) for each sweep
    for tag in args.tags:
        row = {'Sweep': tag}

        # Logit steering
        logit_files = [
            f for f in os.listdir(args.logit_dir)
            if f.endswith('.pkl') and tag in f
        ] if os.path.isdir(args.logit_dir) else []

        ls_meta = None
        if logit_files:
            ls_data = load_pickle(os.path.join(args.logit_dir, logit_files[0]))
            if ls_data and 'metadata' in ls_data:
                ls_meta = ls_data['metadata']
                row['LS N'] = ls_meta.get('n_pairs', '')
                row['LS Base LD'] = f"{ls_meta['mean_baseline_ld']:.3f}"
                row['LS Steer LD'] = f"{ls_meta['mean_steered_ld']:.3f}"
                row['LS Delta LD'] = f"{ls_meta['mean_delta_ld']:.3f}"
            else:
                for k in ['LS N', 'LS Base LD', 'LS Steer LD', 'LS Delta LD']:
                    row[k] = '-'
        else:
            for k in ['LS N', 'LS Base LD', 'LS Steer LD', 'LS Delta LD']:
                row[k] = '-'

        # Build sweep parameter string from logit steering metadata
        if ls_meta:
            layers = ls_meta.get('layers', [])
            offsets = ls_meta.get('offsets', [])
            scale = ls_meta.get('scale', '')
            match_mean = ls_meta.get('match_mean', False)
            alpha = ls_meta.get('alpha', 0.0)
            parts = []
            if layers:
                if len(layers) == 1:
                    parts.append(f"layer={layers[0]}")
                else:
                    parts.append(f"layers={layers[0]}-{layers[-1]}")
            if offsets:
                parts.append(f"offsets={offsets}")
            if match_mean:
                parts.append("match_mean")
                if alpha:
                    parts.append(f"alpha={alpha}")
            elif scale:
                parts.append(f"scale={scale}")
            sweep_params.append((tag, ", ".join(parts)))
        else:
            sweep_params.append((tag, ""))

        # CoT steering
        cot_files = [
            f for f in os.listdir(args.cot_dir)
            if f.endswith('.pkl') and tag in f and 'intervened' in f
        ] if os.path.isdir(args.cot_dir) else []

        if cot_files:
            cot_data = load_pickle(os.path.join(args.cot_dir, cot_files[0]))
            if cot_data and 'successful_results' in cot_data:
                results = cot_data['successful_results']
                counts, total = parse_answers(results)
                if total > 0:
                    row['CoT N'] = total
                    row['CoT Correct'] = f"{counts['correct']/total*100:.1f}%"
                    target_pct = counts['target']/total*100
                    fp = counts.get('target_fp', 0)
                    if fp > 0:
                        true_target = counts['target'] - fp
                        row['CoT Target'] = f"{target_pct:.1f}% ({true_target}+{fp}fp)"
                    else:
                        row['CoT Target'] = f"{target_pct:.1f}%"
                    row['CoT Other'] = f"{counts['other']/total*100:.1f}%"
                    row['CoT Unclear'] = f"{counts['unclear']/total*100:.1f}%"
                else:
                    row['CoT N'] = 0
                    for k in ['CoT Correct', 'CoT Target', 'CoT Other', 'CoT Unclear']:
                        row[k] = '-'
            else:
                row['CoT N'] = '-'
                for k in ['CoT Correct', 'CoT Target', 'CoT Other', 'CoT Unclear']:
                    row[k] = '-'
        else:
            row['CoT N'] = '-'
            for k in ['CoT Correct', 'CoT Target', 'CoT Other', 'CoT Unclear']:
                row[k] = '-'

        rows.append(row)

    headers = [
        'Sweep',
        'LS N', 'LS Base LD', 'LS Steer LD', 'LS Delta LD',
        'CoT N', 'CoT Correct', 'CoT Target', 'CoT Other', 'CoT Unclear',
    ]

    lines.append("RESULTS")
    lines.append("-" * 40)
    lines.append("LS = Logit Steering (Base/Steer/Delta LD = mean logit diff, correct - target)")
    lines.append("CoT = Chain-of-Thought Steering (% of parsed responses)")
    lines.append("")
    lines.append("Sweep parameters:")
    for tag, params in sweep_params:
        lines.append(f"  {tag}: {params}" if params else f"  {tag}: (no metadata)")
    lines.append("")
    lines.append(format_table(rows, headers))
    lines.append("")

    # Baseline-correct-only table: steered results filtered to datapoints correct in baseline
    baseline_correct_indices = _get_baseline_correct_indices(args)
    if baseline_correct_indices is not None:
        filtered_rows = []
        for tag in args.tags:
            if tag == "baseline":
                continue
            row = {'Sweep': tag}
            cot_files = [
                f for f in os.listdir(args.cot_dir)
                if f.endswith('.pkl') and tag in f and 'intervened' in f
            ] if os.path.isdir(args.cot_dir) else []

            if cot_files:
                cot_data = load_pickle(os.path.join(args.cot_dir, cot_files[0]))
                if cot_data and 'successful_results' in cot_data:
                    filtered = [r for r in cot_data['successful_results']
                                if r.get('batch_idx') in baseline_correct_indices]
                    counts, total = parse_answers(filtered)
                    if total > 0:
                        row['N'] = total
                        row['Correct'] = f"{counts['correct']/total*100:.1f}%"
                        target_pct = counts['target']/total*100
                        fp = counts.get('target_fp', 0)
                        if fp > 0:
                            true_target = counts['target'] - fp
                            row['Target'] = f"{target_pct:.1f}% ({true_target}+{fp}fp)"
                        else:
                            row['Target'] = f"{target_pct:.1f}%"
                        row['Other'] = f"{counts['other']/total*100:.1f}%"
                        row['Unclear'] = f"{counts['unclear']/total*100:.1f}%"
                    else:
                        row['N'] = 0
                        for k in ['Correct', 'Target', 'Other', 'Unclear']:
                            row[k] = '-'
                else:
                    for k in ['N', 'Correct', 'Target', 'Other', 'Unclear']:
                        row[k] = '-'
            else:
                for k in ['N', 'Correct', 'Target', 'Other', 'Unclear']:
                    row[k] = '-'
            filtered_rows.append(row)

        if filtered_rows:
            filtered_headers = ['Sweep', 'N', 'Correct', 'Target', 'Other', 'Unclear']
            lines.append("COT STEERING (BASELINE-CORRECT ONLY)")
            lines.append("-" * 40)
            lines.append(f"Filtered to {len(baseline_correct_indices)} datapoints "
                         f"that the model answered correctly without steering.")
            lines.append("")
            lines.append(format_table(filtered_rows, filtered_headers))
            lines.append("")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logit_dir', required=True)
    parser.add_argument('--cot_dir', required=True)
    parser.add_argument('--patching_dir', default='')
    parser.add_argument('--projections_path', default='')  # unused, kept for CLI compat
    parser.add_argument('--tags', nargs='+', required=True)
    parser.add_argument('--output_file', default=None,
                        help='Path to save summary file')
    # Hyperparameter args (all optional, for display only)
    parser.add_argument('--model_name', default='')
    parser.add_argument('--suffix', default='')
    parser.add_argument('--num_datapoints', default='')
    parser.add_argument('--min_confidence', default='')
    parser.add_argument('--min_choice_tokens', default='')
    parser.add_argument('--max_choice_tokens', default='')
    parser.add_argument('--seed', default='')
    parser.add_argument('--test_fraction', default='')
    parser.add_argument('--direction_offsets', default='')
    parser.add_argument('--patch_mode', default='')
    parser.add_argument('--patch_offsets', default='')
    args = parser.parse_args()

    summary = build_summary(args)

    # Print to stdout
    print(flush=True)
    print(summary, flush=True)

    # Save to file
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
        with open(args.output_file, 'w') as f:
            f.write(summary + '\n')
        print(f"Summary saved to: {args.output_file}", flush=True)


if __name__ == '__main__':
    main()
