#!/usr/bin/env python3
"""Run correctness direction steering on arbitrary instruct models.

Generates MMLU responses while intervening on the model's internal
representations at answer delimiter positions. The intervention:
- Subtracts scale * correctness_direction from the correct answer's positions
- Adds scale * correctness_direction to a random incorrect answer's positions

This aims to flip the model's output toward the incorrect (target) answer.

Usage:
    # Generate with intervention
    python run_steering.py --stage generate \\
        --model_name Qwen/Qwen2.5-7B-Instruct \\
        --layers 15 16 17 18 19 20 \\
        --scale 50.0 --num_datapoints 100 --suffix scale50

    # Inspect results
    python run_steering.py --stage inspect --suffix scale50

    # Compare with a baseline
    python run_steering.py --stage compare --suffix scale50 \\
        --baseline_path /path/to/baseline_generations.pkl

Stages:
    generate - Run intervened generation (requires GPU)
    inspect  - Analyse the generated results
    compare  - Compare intervened vs baseline results
"""

import argparse
import os
import pickle
import re
import string
import sys
from pathlib import Path
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from src.direction_finder import model_slug
from src.summarize_results import extract_answer, is_false_positive

# Paths
EXP_DIR = Path(SCRIPT_DIR)
OUTPUT_DIR = EXP_DIR / "output"  # created on demand in get_paths


def get_paths(model_name: str, suffix: str = "", output_dir: str = None):
    """Get file paths with optional suffix, namespaced by model."""
    if output_dir:
        model_dir = Path(output_dir)
    else:
        slug = model_slug(model_name)
        model_dir = OUTPUT_DIR / slug
    model_dir.mkdir(parents=True, exist_ok=True)
    suffix_str = f"_{suffix}" if suffix else ""
    return {
        'intervened': model_dir / f"intervened_generations{suffix_str}.pkl",
    }


def default_direction_path(model_name: str) -> str:
    """Return default direction path: output/{model_slug}/direction.pt"""
    slug = model_slug(model_name)
    return str(EXP_DIR / "output" / slug / "direction.pt")


def load_patching_pairs_as_steering_data(data_path: str, model_name: str,
                                         enable_thinking: bool = True):
    """Load patching pairs pickle and convert to steering format.

    Reformats each item's question with the same answer
    ordering for CoT generation (add_generation_prompt=True).

    Returns:
        List of (prompt, correct_idx, answer_texts, debug_info) tuples
        where debug_info is a dict with original pair metadata for debugging.
    """
    from transformers import AutoTokenizer

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    pairs = data['pairs']
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Default dataset from metadata ('dataset' or first of 'datasets'), fall back
    # to mmlu; per-pair 'source_dataset' overrides for mixed-dataset runs
    from src.data_generator import DATASET_INSTRUCTIONS
    meta = data.get('metadata', {})
    default_dataset = meta.get('dataset') or (meta.get('datasets') or ['mmlu'])[0]
    steering_data = []

    for pair in pairs:
        dataset = pair.get('source_dataset') or default_dataset
        instruction = DATASET_INSTRUCTIONS.get(dataset, DATASET_INSTRUCTIONS['mmlu'])
        question_prepend = f'{instruction} Think step-by-step before giving your answer. \n'
        question = pair['question']
        n_choices = len(pair['order'])
        letters = list(string.ascii_lowercase[:n_choices])
        answer_texts = [pair['answer_texts'][pair['order'][i]] for i in range(n_choices)]
        correct_idx = pair['correct_pos']

        # Format for generation (same as get_mmlu_dataloader)
        labeled_choices = [f"({letters[i]}) {answer_texts[i]}" for i in range(n_choices)]
        formatted_q = question + '\nChoices: ' + ', '.join(labeled_choices) + '.\n'
        messages = [{"role": "user", "content": question_prepend + formatted_q}]
        template_kwargs = {}
        if not enable_thinking:
            # Qwen3/Qwen3.6: renders an empty <think>\n\n</think>\n\n so the
            # model answers directly. Templates without the variable ignore it.
            template_kwargs['enable_thinking'] = False
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_kwargs
        )

        # Include debug metadata
        debug_info = {
            'question': question,
            'subject': pair.get('subject', ''),
            'order': pair['order'],
            'correct_pos': pair['correct_pos'],
            'original_answer_texts': pair['answer_texts'],
        }

        steering_data.append((prompt, correct_idx, answer_texts, debug_info))

    print(f"Loaded {len(steering_data)} examples from patching pairs: {data_path}")
    return steering_data


def stage_generate(
    num_datapoints: int,
    scale: float,
    layers: list,
    offsets: list,
    suffix: str,
    direction_path: str,
    model_name: str,
    do_sample: bool = False,
    seed: int = 42,
    debug: bool = False,
    normal_noise: bool = False,
    noise_scale: float = 100.0,
    batch_size: int = 1,
    data_path: str = None,
    match_mean: bool = False,
    mean_match_std_dev: float = 0.0,
    projections_path: str = None,
    min_choice_tokens: int = None,
    max_choice_tokens: int = None,
    require_equal_tokens: bool = False,
    mean_matching_debug: bool = False,
    max_new_tokens: int = 1000,
    cell_mask_path: str = None,
    enable_thinking: bool = True,
    output_dir: str = None,
    adaptive_offsets: bool = False,
    dataset: str = 'mmlu',
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    fp32: bool = False,
    cache_dir: str = None,
):
    """Run intervened generation."""
    print("=" * 60)
    print("STAGE: INTERVENED GENERATION")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Direction: {direction_path}")
    print(f"Num datapoints: {num_datapoints}")
    print(f"Mode: {'match_mean' if match_mean else 'fixed_scale'}")
    if match_mean:
        print(f"Projections: {projections_path}")
        if mean_match_std_dev != 0.0:
            print(f"Std dev multiplier: {mean_match_std_dev}")
    else:
        print(f"Scale: {scale}")
    print(f"Layers: {layers}")
    print(f"Offsets: {offsets}")
    print(f"Suffix: {suffix or '(none)'}")
    print(f"Data path: {data_path or '(MMLU)'}")
    print()

    from src.intervened_generator import generate_with_intervention

    paths = get_paths(model_name, suffix, output_dir=output_dir)

    # Load preformatted data from patching pairs if provided
    preloaded_data = None
    if data_path is not None:
        preloaded_data = load_patching_pairs_as_steering_data(data_path, model_name,
                                                              enable_thinking=enable_thinking)

    # Token length filtering on preloaded data
    has_filter = (min_choice_tokens is not None or
                  max_choice_tokens is not None or
                  require_equal_tokens)
    if preloaded_data is not None and has_filter:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        n_before = len(preloaded_data)
        filtered = []
        for item in preloaded_data:
            # Handle both 3-tuple and 4-tuple formats
            prompt, correct_idx, answer_texts = item[0], item[1], item[2]
            token_lengths = [
                len(tokenizer.encode(" " + c, add_special_tokens=False))
                for c in answer_texts
            ]
            if min_choice_tokens is not None and any(
                    tl < min_choice_tokens for tl in token_lengths):
                continue
            if max_choice_tokens is not None and any(
                    tl > max_choice_tokens for tl in token_lengths):
                continue
            if require_equal_tokens and len(set(token_lengths)) != 1:
                continue
            filtered.append(item)  # Keep full tuple including debug_info
        preloaded_data = filtered
        print(f"Token length filter: {n_before} -> {len(preloaded_data)} examples")

    cell_mask = None
    if cell_mask_path:
        import json
        with open(cell_mask_path) as f:
            cell_mask = {int(k): v for k, v in json.load(f).items()}
        print(f"Loaded cell mask from {cell_mask_path}: {len(cell_mask)} layers")

    result = generate_with_intervention(
        model_name=model_name,
        direction_path=direction_path,
        output_path=str(paths['intervened']),
        num_datapoints=num_datapoints,
        intervention_layers=layers,
        intervention_offsets=offsets,
        scale=scale,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        cell_mask=cell_mask,
        do_sample=do_sample,
        seed=seed,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        fp32=fp32,
        cache_dir=cache_dir,
        debug=debug,
        normal_noise=normal_noise,
        noise_scale=noise_scale,
        preloaded_data=preloaded_data,
        match_mean=match_mean,
        mean_match_std_dev=mean_match_std_dev,
        projections_path=projections_path,
        mean_matching_debug=mean_matching_debug,
        adaptive_offsets=adaptive_offsets,
        dataset=dataset,
    )

    return result


def stage_inspect(model_name: str, suffix: str, output_dir: str = None):
    """Inspect generation results."""
    print("=" * 60)
    print("INSPECTING INTERVENED GENERATION RESULTS")
    print("=" * 60)

    paths = get_paths(model_name, suffix, output_dir=output_dir)

    with open(paths['intervened'], 'rb') as f:
        data = pickle.load(f)

    print(f"\nMetadata:")
    for key, value in data['metadata'].items():
        print(f"  {key}: {value}")

    results = data['results']
    successful = data['successful_results']

    print(f"\nTotal results: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(results) - len(successful)}")

    if len(successful) == 0:
        print("\nNo successful generations to analyze.")
        return

    # Analyze target selection
    target_counts = Counter(r['target_idx'] for r in successful)
    all_indices = set(r['target_idx'] for r in successful) | set(r['correct_idx'] for r in successful)
    print(f"\nTarget answer distribution:")
    for idx in sorted(all_indices):
        letter = string.ascii_lowercase[idx]
        count = target_counts.get(idx, 0)
        print(f"  ({letter}): {count} ({100*count/len(successful):.1f}%)")

    # Check for answer mentions in responses
    print(f"\nAnswer analysis (checking final answer in response):")
    n_choices = max(r.get('target_idx', 0) for r in successful) + 1
    n_choices = max(n_choices, max(r.get('correct_idx', 0) for r in successful) + 1)
    letters = list(string.ascii_lowercase[:n_choices])

    answered_correct = 0
    answered_target = 0
    answered_target_fp = 0
    answered_other = 0
    unclear = 0

    # Thinking-model run if any response closed a think block (or reached the
    # Harmony final channel); then responses without one are truncated
    # mid-reasoning and must not be parsed.
    is_thinking_run = any(('</think>' in (r.get('response') or '')) or ('assistantfinal' in (r.get('response') or ''))
                          for r in successful)

    for r in successful:
        correct_letter = letters[r['correct_idx']]
        target_letter = letters[r['target_idx']]
        answer_texts = r.get('answer_texts')

        found_idx = extract_answer(r['response'], letters, answer_texts,
                                   require_think_close=is_thinking_run)
        found = letters[found_idx] if found_idx is not None and found_idx < len(letters) else None

        if found is None:
            unclear += 1
        elif found == correct_letter:
            answered_correct += 1
        elif found == target_letter:
            answered_target += 1
            if is_false_positive(r['response'], found_idx, r['correct_idx'], r['target_idx'], answer_texts):
                answered_target_fp += 1
        else:
            answered_other += 1

    n = len(successful)
    print(f"  Answered correct: {answered_correct} ({100*answered_correct/n:.1f}%)")
    print(f"  Answered target (desired flip): {answered_target} ({100*answered_target/n:.1f}%)")
    if answered_target_fp > 0:
        true_target = answered_target - answered_target_fp
        print(f"    True target (letter + text match): {true_target} ({100*true_target/n:.1f}%)")
        print(f"    False positive (target letter, correct text): {answered_target_fp} ({100*answered_target_fp/n:.1f}%)")
    print(f"  Answered other: {answered_other} ({100*answered_other/n:.1f}%)")
    print(f"  Unclear/unparseable: {unclear} ({100*unclear/n:.1f}%)")

    # Show some examples
    print(f"\n{'='*60}")
    print("EXAMPLE GENERATIONS")
    print(f"{'='*60}")

    for i, r in enumerate(successful[:3]):
        print(f"\n--- Example {i} ---")
        print(f"Correct answer: ({letters[r['correct_idx']]})")
        print(f"Target answer (intervention): ({letters[r['target_idx']]})")
        print(f"Answer texts: {r['answer_texts']}")
        print(f"\nResponse (last 500 chars):")
        print(r['response'][-500:])
        print()


def _parse_answers(responses, correct_indices):
    """Parse final answers from a list of responses and compute accuracy stats."""
    n_choices = max(correct_indices) + 1
    letters = list(string.ascii_lowercase[:n_choices])
    counts = {'correct': 0, 'incorrect': 0, 'unclear': 0}

    is_thinking_run = any(('</think>' in (r or '')) or ('assistantfinal' in (r or '')) for r in responses)

    for i, response in enumerate(responses):
        correct_letter = letters[correct_indices[i]]
        found_idx = extract_answer(response, letters,
                                   require_think_close=is_thinking_run)
        found = letters[found_idx] if found_idx is not None else None

        if found is None:
            counts['unclear'] += 1
        elif found == correct_letter:
            counts['correct'] += 1
        else:
            counts['incorrect'] += 1

    return counts


def stage_compare(model_name: str, suffix: str, baseline_path: str, output_dir: str = None):
    """Compare intervened vs baseline results."""
    print("=" * 60)
    print("COMPARING INTERVENED VS BASELINE")
    print("=" * 60)

    paths = get_paths(model_name, suffix, output_dir=output_dir)

    # Load intervened data
    try:
        with open(paths['intervened'], 'rb') as f:
            intervened_data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Intervened results not found at {paths['intervened']}")
        print("Run the 'generate' stage first.")
        return

    # Load baseline data
    try:
        with open(baseline_path, 'rb') as f:
            baseline_data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Baseline not found at {baseline_path}")
        return

    intervened = intervened_data['successful_results']

    # Support both our own format (results/successful_results) and the cot_generator format (generations/answers)
    if 'successful_results' in baseline_data:
        baseline_responses = [r['response'] for r in baseline_data['successful_results']]
        baseline_correct = [r['correct_idx'] for r in baseline_data['successful_results']]
    elif 'generations' in baseline_data:
        baseline_responses = baseline_data['generations']
        baseline_correct = baseline_data['answers']
    else:
        print(f"Error: Unrecognised baseline format. Keys: {list(baseline_data.keys())}")
        return

    print(f"\nIntervened: {len(intervened)} examples")
    print(f"Baseline: {len(baseline_responses)} examples")

    # Baseline analysis
    baseline_counts = _parse_answers(baseline_responses, baseline_correct)
    n_baseline = len(baseline_responses)
    print(f"\nBaseline accuracy:")
    print(f"  Correct: {baseline_counts['correct']} ({100*baseline_counts['correct']/n_baseline:.1f}%)")
    print(f"  Incorrect: {baseline_counts['incorrect']} ({100*baseline_counts['incorrect']/n_baseline:.1f}%)")
    print(f"  Unclear: {baseline_counts['unclear']} ({100*baseline_counts['unclear']/n_baseline:.1f}%)")

    # Intervened analysis
    intervened_responses = [r['response'] for r in intervened]
    intervened_correct_indices = [r['correct_idx'] for r in intervened]
    intervened_counts = _parse_answers(intervened_responses, intervened_correct_indices)
    n_intervened = len(intervened)

    print(f"\nIntervened accuracy:")
    print(f"  Correct: {intervened_counts['correct']} ({100*intervened_counts['correct']/n_intervened:.1f}%)")
    print(f"  Incorrect: {intervened_counts['incorrect']} ({100*intervened_counts['incorrect']/n_intervened:.1f}%)")
    print(f"  Unclear: {intervened_counts['unclear']} ({100*intervened_counts['unclear']/n_intervened:.1f}%)")

    # Change in accuracy
    baseline_acc = baseline_counts['correct'] / n_baseline
    intervened_acc = intervened_counts['correct'] / n_intervened
    delta = intervened_acc - baseline_acc

    print(f"\nAccuracy change: {delta*100:+.1f}%")
    if delta < 0:
        print("  (Intervention successfully reduced accuracy toward correct answer)")


def main():
    parser = argparse.ArgumentParser(
        description="Correctness direction steering (model-agnostic)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--stage", type=str, required=True,
                        choices=['generate', 'inspect', 'compare'],
                        help="Which stage to run")
    parser.add_argument("--model_name", type=str, required=True,
                        help="HuggingFace model ID (e.g. Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix for output files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: output/{model_slug})")

    # Generation parameters
    parser.add_argument("--num_datapoints", type=int, default=100,
                        help="Number of examples (default: 100)")
    parser.add_argument("--scale", type=float, default=None,
                        help="Intervention scale (0 for baseline, ignored in match_mean mode)")
    parser.add_argument("--layers", type=int, nargs='+', default=None,
                        help="Intervention layers")
    parser.add_argument("--all_layers", action="store_true",
                        help="Steer at all layers (resolved via model config)")
    parser.add_argument("--offsets", type=int, nargs='+', default=[-1, 0, 1],
                        help="Position offsets (default: -1 0 1)")
    parser.add_argument("--direction_path", type=str, default=None,
                        help="Path to direction tensor (default: output/{model_slug}/direction.pt)")
    parser.add_argument("--match_mean", action="store_true",
                        help="Match-mean mode: steer projections to opposite class mean (ignores --scale)")
    parser.add_argument("--mean_match_std_dev", type=float, default=0.0,
                        help="In match_mean mode, steer correct to incorrect_mean - n*std, "
                             "target to correct_mean + n*std (default: 0, just use means)")
    parser.add_argument("--projections_path", type=str, default=None,
                        help="Path to projections.pkl for match_mean mode "
                             "(default: output/{model_slug}/projections.pkl)")
    parser.add_argument("--min_choice_tokens", type=int, default=None,
                        help="Min tokens per answer choice (default: no filter)")
    parser.add_argument("--max_choice_tokens", type=int, default=None,
                        help="Max tokens per answer choice (default: no filter)")
    parser.add_argument("--require_equal_tokens", action="store_true",
                        help="Only keep pairs where all 4 choices have equal token length")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--do_sample", action="store_true",
                        help="Use sampling instead of greedy decoding")
    parser.add_argument("--debug", action="store_true",
                        help="Print each generation as it completes")
    parser.add_argument("--mean_matching_debug", action="store_true",
                        help="In match_mean mode, print tables of projections, "
                             "targets, and steering deltas for each example")
    parser.add_argument("--normal_noise", action="store_true",
                        help="TEST MODE: Add random noise to all positions instead of direction intervention")
    parser.add_argument("--noise_scale", type=float, default=100.0,
                        help="Scale of random noise when using --normal_noise (default: 100.0)")
    parser.add_argument("--max_new_tokens", type=int, default=1000,
                        help="Max new tokens per generation (raise for thinking models)")
    parser.add_argument("--cell_mask_path", type=str, default=None,
                        help="JSON file {layer: [offset values]} restricting steering to those (layer, offset) cells")
    parser.add_argument("--nothink", action="store_true",
                        help="Disable thinking mode (enable_thinking=False) in generation prompts")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for generation (default: 4)")
    parser.add_argument("--adaptive_offsets", action="store_true",
                        help="Adaptive offsets: clip per-position offsets to answer token length")

    # Model loading
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Load model in 8-bit quantization")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load model in 4-bit quantization (NF4)")
    parser.add_argument("--fp32", action="store_true",
                        help="Load model in float32 (more precise, more memory)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HuggingFace model cache directory")

    # Data source
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to patching pairs pickle (filtered or raw). "
                             "When provided, uses these questions instead of sampling fresh.")
    parser.add_argument("--dataset", default="mmlu", choices=["mmlu", "hellaswag", "truthfulqa"],
                        help="Dataset to use when sampling fresh (default: mmlu)")

    # Compare parameters
    parser.add_argument("--baseline_path", type=str, default=None,
                        help="Path to baseline generations pickle (for compare stage)")

    args = parser.parse_args()

    # Default direction path based on model name
    if args.direction_path is None:
        args.direction_path = default_direction_path(args.model_name)

    # Validate layers
    if args.stage == 'generate':
        if not args.all_layers and args.layers is None:
            parser.error("either --layers or --all_layers is required for generate stage")
        if not args.match_mean and args.scale is None:
            parser.error("--scale is required unless --match_mean is used")

    # Resolve --all_layers via model config (no GPU needed)
    if args.all_layers:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.model_name)
        n_layers = getattr(config, 'num_hidden_layers', None) or config.text_config.num_hidden_layers
        args.layers = list(range(n_layers))
        print(f"All layers: {len(args.layers)} layers")

    # Resolve projections path for match_mean mode
    if args.match_mean:
        if args.projections_path is None:
            slug = model_slug(args.model_name)
            args.projections_path = os.path.join(SCRIPT_DIR, 'output', slug, 'projections.pkl')
        if not os.path.exists(args.projections_path):
            parser.error(f"Match-mean mode requires projections.pkl, not found at: {args.projections_path}")

    if args.stage == 'generate':
        stage_generate(
            num_datapoints=args.num_datapoints,
            scale=args.scale,
            layers=args.layers,
            offsets=args.offsets,
            suffix=args.suffix,
            direction_path=args.direction_path,
            model_name=args.model_name,
            do_sample=args.do_sample,
            seed=args.seed,
            debug=args.debug,
            normal_noise=args.normal_noise,
            noise_scale=args.noise_scale,
            batch_size=args.batch_size,
            data_path=args.data_path,
            match_mean=args.match_mean,
            mean_match_std_dev=args.mean_match_std_dev,
            projections_path=args.projections_path,
            min_choice_tokens=args.min_choice_tokens,
            max_choice_tokens=args.max_choice_tokens,
            require_equal_tokens=args.require_equal_tokens,
            mean_matching_debug=args.mean_matching_debug,
            max_new_tokens=args.max_new_tokens,
            cell_mask_path=args.cell_mask_path,
            enable_thinking=not args.nothink,
            output_dir=args.output_dir,
            adaptive_offsets=args.adaptive_offsets,
            dataset=args.dataset,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            fp32=args.fp32,
            cache_dir=args.cache_dir,
        )
    elif args.stage == 'inspect':
        stage_inspect(model_name=args.model_name, suffix=args.suffix, output_dir=args.output_dir)
    elif args.stage == 'compare':
        if args.baseline_path is None:
            parser.error("--baseline_path is required for the compare stage")
        stage_compare(
            model_name=args.model_name,
            suffix=args.suffix,
            baseline_path=args.baseline_path,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
