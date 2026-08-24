#!/usr/bin/env python3
"""Final evaluation pipeline: direction finding, patching, steering, and analysis.

Runs all stages in one sweep with train/dev/test split:
  - Train: direction finding, patching, BC position means
  - Dev: layer sweeps (identify optimal layers)
  - Test: logit steering, CoT steering, BC accuracy/correlation

Reports results combined and per-dataset.

Usage:
    python final_pipeline/run_full_pipeline.py --config final_pipeline/default_config.yaml --output_dir runs/qwen_mixed
    python final_pipeline/run_full_pipeline.py --config final_pipeline/default_config.yaml --output_dir runs/qwen_mixed --stages filter direction
"""

import argparse
import gc
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PIPELINE_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def log(msg):
    print(msg, flush=True)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pkl(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    log(f"Saved {path}")


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def load_steering_flip_rate(path):
    """Load steering result and return (n, flip_rate, mean_delta_ld)."""
    if not os.path.exists(path):
        return None
    r = load_pkl(path)
    s = r['steered_ld'].numpy() if torch.is_tensor(r['steered_ld']) else np.array(r['steered_ld'])
    d = r['delta_ld'].numpy() if torch.is_tensor(r['delta_ld']) else np.array(r['delta_ld'])
    v = ~np.isnan(s)
    n = v.sum()
    if n == 0:
        return None
    return n, (s[v] < 0).sum() / n, np.nanmean(d)


# ---------------------------------------------------------------------------
# Stage 1: Filter and Split
# ---------------------------------------------------------------------------

def stage_filter_and_split(cfg, output_dir):
    """Generate filtered pairs from each dataset, merge, and split."""
    log("\n" + "=" * 60)
    log("  STAGE 1: FILTER AND SPLIT")
    log("=" * 60)

    from src.hf_utils import load_model_hf
    from src.data_generator_hf import generate_and_filter_pairs_hf
    from src.direction_finder import _NEGATION_WORDS

    bundle = load_model_hf(
        cfg['model_name'],
        load_in_8bit=cfg.get('load_in_8bit', False),
        load_in_4bit=cfg.get('load_in_4bit', False),
        cache_dir=cfg.get('cache_dir'),
    )

    target = cfg['num_datapoints_per_dataset']
    skip_negated = cfg['filtering']['skip_negated']

    all_pairs = []
    for dataset in cfg['datasets']:
        log(f"\n  Generating up to {target} from {dataset}...")

        # Over-request to account for strict negation filter (~25% loss)
        request_size = int(target * 2.0) if skip_negated else target
        kept = []
        n_strict_removed = 0

        batch = generate_and_filter_pairs_hf(
            bundle=bundle,
            model_name=cfg['model_name'],
            num_datapoints=request_size,
            seed=cfg['seed'],
            min_choice_tokens=cfg['filtering']['min_choice_tokens'],
            max_choice_tokens=cfg['filtering']['max_choice_tokens'],
            min_confidence=cfg['filtering']['min_confidence'],
            require_equal_tokens=False,
            skip_negated=skip_negated,
            dataset=dataset,
            n_choices=cfg['filtering'].get('n_choices'),
        )

        # Apply strict negation filter (checks full prompt, not just question)
        for p in batch['pairs']:
            p['source_dataset'] = dataset
            if skip_negated:
                gen = p.get('gen', '')
                words = set(gen.lower().split())
                if words & _NEGATION_WORDS:
                    n_strict_removed += 1
                    continue
            kept.append(p)

        # Trim to target
        if len(kept) > target:
            kept = kept[:target]

        all_pairs.extend(kept)
        log(f"  Got {len(kept)} {dataset} pairs"
            f" (strict negation removed {n_strict_removed})"
            f"{' [EXHAUSTED DATASET]' if len(kept) < target else ''}")

    del bundle
    free_gpu()

    # Shuffle and split
    random.seed(cfg['seed'])
    random.shuffle(all_pairs)

    n = len(all_pairs)
    train_frac = cfg['split']['train_fraction']
    dev_frac = cfg['split']['dev_fraction']

    n_train = int(n * train_frac)
    n_dev = int(n * dev_frac)

    train_pairs = all_pairs[:n_train]
    dev_pairs = all_pairs[n_train:n_train + n_dev]
    test_pairs = all_pairs[n_train + n_dev:]

    log(f"\n  Total: {n} pairs")
    log(f"  Train: {len(train_pairs)}, Dev: {len(dev_pairs)}, Test: {len(test_pairs)}")

    for split_name, pairs in [('train', train_pairs), ('dev', dev_pairs), ('test', test_pairs)]:
        ds_counts = {}
        for p in pairs:
            ds = p.get('source_dataset', 'unknown')
            ds_counts[ds] = ds_counts.get(ds, 0) + 1
        log(f"    {split_name}: " + ", ".join(f"{k}={v}" for k, v in sorted(ds_counts.items())))

    # Save
    for split_name, pairs in [('train', train_pairs), ('dev', dev_pairs), ('test', test_pairs)]:
        save_pkl(
            {'pairs': pairs, 'metadata': {'split': split_name, 'n': len(pairs),
             'datasets': cfg['datasets'], 'seed': cfg['seed']}},
            os.path.join(output_dir, 'data', f'{split_name}.pkl')
        )

    # Also save full merged set
    save_pkl(
        {'pairs': all_pairs, 'metadata': {'n': n, 'datasets': cfg['datasets']}},
        os.path.join(output_dir, 'data', 'all.pkl')
    )


# ---------------------------------------------------------------------------
# Stage 2: Direction Finding (train set)
# ---------------------------------------------------------------------------

def stage_direction(cfg, output_dir):
    """Extract correctness direction from train set."""
    log("\n" + "=" * 60)
    log("  STAGE 2: DIRECTION FINDING (train)")
    log("=" * 60)

    from src.direction_finder_hf_streaming import extract_and_save_direction_streaming

    extract_and_save_direction_streaming(
        model_name=cfg['model_name'],
        data_path=os.path.join(output_dir, 'data', 'train.pkl'),
        output_dir=os.path.join(output_dir, 'direction'),
        offsets=cfg['direction']['offsets'],
        min_confidence=0,
        skip_negated=False,  # Already filtered in stage 1
        adaptive_offsets=cfg['direction']['adaptive_offsets'],
        load_in_8bit=cfg.get('load_in_8bit', False),
        load_in_4bit=cfg.get('load_in_4bit', False),
        cache_dir=cfg.get('cache_dir'),
    )

    # Create BC projections
    from final_pipeline.create_bc_projections_lib import create_bc_projections
    create_bc_projections(
        os.path.join(output_dir, 'direction', 'projections.pkl'),
        os.path.join(output_dir, 'direction', 'projections_bc.pkl'),
    )


# ---------------------------------------------------------------------------
# Stage 3: Patching (train set)
# ---------------------------------------------------------------------------

def stage_patching(cfg, output_dir):
    """Run activation patching on train set."""
    log("\n" + "=" * 60)
    log("  STAGE 3: PATCHING (train)")
    log("=" * 60)

    from src.hf_utils import load_model_hf
    from src.activation_patching_hf import run_random_swap_patching_hf
    from src.activation_patching import plot_patching_heatmap

    train_data = load_pkl(os.path.join(output_dir, 'data', 'train.pkl'))

    # Subsample for patching, balanced across datasets
    max_pairs = cfg['patching'].get('max_pairs')
    if max_pairs and len(train_data['pairs']) > max_pairs:
        pairs = train_data['pairs']
        datasets = cfg.get('datasets', [])
        if len(datasets) > 1:
            per_ds = max_pairs // len(datasets)
            by_ds = {ds: [] for ds in datasets}
            other = []
            for p in pairs:
                ds = p.get('source_dataset', '')
                if ds in by_ds:
                    by_ds[ds].append(p)
                else:
                    other.append(p)
            random.seed(cfg['seed'])
            sampled = []
            for ds in datasets:
                pool = by_ds[ds]
                random.shuffle(pool)
                sampled.extend(pool[:per_ds])
            # Fill remainder from largest pools
            remainder = max_pairs - len(sampled)
            if remainder > 0:
                leftover = []
                for ds in datasets:
                    leftover.extend(by_ds[ds][per_ds:])
                leftover.extend(other)
                random.shuffle(leftover)
                sampled.extend(leftover[:remainder])
            random.shuffle(sampled)
            train_data = {**train_data, 'pairs': sampled}
            ds_counts = {}
            for p in sampled:
                ds = p.get('source_dataset', 'unknown')
                ds_counts[ds] = ds_counts.get(ds, 0) + 1
            log(f"  Patching subsample: {len(sampled)} pairs ({', '.join(f'{k}={v}' for k, v in sorted(ds_counts.items()))})")
        else:
            random.seed(cfg['seed'])
            random.shuffle(pairs)
            train_data = {**train_data, 'pairs': pairs[:max_pairs]}
            log(f"  Patching subsample: {max_pairs} pairs")

    bundle = load_model_hf(
        cfg['model_name'],
        load_in_8bit=cfg.get('load_in_8bit', False),
        load_in_4bit=cfg.get('load_in_4bit', False),
        cache_dir=cfg.get('cache_dir'),
    )

    patch_dir = os.path.join(output_dir, 'patching')
    os.makedirs(patch_dir, exist_ok=True)

    # Per-offset patching
    result = run_random_swap_patching_hf(
        bundle=bundle, data=train_data,
        offsets=cfg['patching']['offsets'],
        seed=cfg['seed'],
        batch_size=cfg['patching']['batch_size'],
        adaptive_offsets=cfg['direction']['adaptive_offsets'],
    )
    save_pkl(result, os.path.join(patch_dir, 'patching_per_offset.pkl'))

    # All-offsets patching
    result_alloff = run_random_swap_patching_hf(
        bundle=bundle, data=train_data,
        offsets=cfg['patching']['offsets'],
        seed=cfg['seed'],
        batch_size=cfg['patching']['batch_size'],
        adaptive_offsets=cfg['direction']['adaptive_offsets'],
        all_offsets=True,
    )
    save_pkl(result_alloff, os.path.join(patch_dir, 'patching_all_offsets.pkl'))

    del bundle
    free_gpu()


# ---------------------------------------------------------------------------
# Stage 4: BC Accuracy/Correlation Heatmaps (test set projections)
# ---------------------------------------------------------------------------

def stage_test_projections(cfg, output_dir):
    """Extract projections on test set using train direction, generate BC heatmaps."""
    log("\n" + "=" * 60)
    log("  STAGE 4: TEST PROJECTIONS + BC HEATMAPS")
    log("=" * 60)

    from src.direction_finder_hf_streaming import extract_and_save_direction_streaming

    direction_path = os.path.join(output_dir, 'direction', 'direction.pt')
    test_proj_dir = os.path.join(output_dir, 'test_projections')
    os.makedirs(test_proj_dir, exist_ok=True)

    extract_and_save_direction_streaming(
        model_name=cfg['model_name'],
        data_path=os.path.join(output_dir, 'data', 'test.pkl'),
        output_dir=test_proj_dir,
        offsets=cfg['direction']['offsets'],
        min_confidence=0,
        skip_negated=False,  # Already filtered in stage 1
        existing_direction_path=direction_path,
        adaptive_offsets=cfg['direction']['adaptive_offsets'],
        load_in_8bit=cfg.get('load_in_8bit', False),
        load_in_4bit=cfg.get('load_in_4bit', False),
        cache_dir=cfg.get('cache_dir'),
    )

    # Generate BC heatmaps for test set using train position means
    _generate_bc_heatmaps(cfg, output_dir, 'test')


# ---------------------------------------------------------------------------
# Stage 5: Layer Sweeps (dev set)
# ---------------------------------------------------------------------------

def stage_layer_sweeps(cfg, output_dir):
    """Run per-layer and window sweeps on dev set to identify optimal layers."""
    log("\n" + "=" * 60)
    log("  STAGE 5: LAYER SWEEPS (dev)")
    log("=" * 60)

    from src.hf_utils import load_model_hf
    from src.logit_steering_hf_batched import run_logit_steering_hf
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(cfg['model_name'])
    n_layers = getattr(config, 'num_hidden_layers', None) or config.text_config.num_hidden_layers
    direction_path = os.path.join(output_dir, 'direction', 'direction.pt')
    projections_path = os.path.join(output_dir, 'direction', 'projections.pkl')
    dev_data_path = os.path.join(output_dir, 'data', 'dev.pkl')

    if not load_pkl(dev_data_path)['pairs']:
        log("  Dev set is empty — skipping layer sweeps (steering will use all layers)")
        return

    sweep_dir = os.path.join(output_dir, 'layer_sweeps')
    os.makedirs(os.path.join(sweep_dir, 'per_layer'), exist_ok=True)
    os.makedirs(os.path.join(sweep_dir, 'window'), exist_ok=True)

    bundle = load_model_hf(
        cfg['model_name'],
        load_in_8bit=cfg.get('load_in_8bit', False),
        load_in_4bit=cfg.get('load_in_4bit', False),
        cache_dir=cfg.get('cache_dir'),
    )

    dev_data = load_pkl(dev_data_path)
    n_sweep = cfg['layer_sweeps']['num_datapoints']

    # Per-layer sweep
    log(f"\n  Per-layer sweep ({n_layers} layers, {n_sweep} examples)...")
    for layer in range(n_layers):
        out_path = os.path.join(sweep_dir, 'per_layer', f'L{layer}.pkl')
        if os.path.exists(out_path):
            continue
        result = run_logit_steering_hf(
            bundle=bundle, data=dev_data,
            direction_path=direction_path,
            layers=[layer], offsets=cfg['steering']['offsets'],
            scale=0, seed=cfg['seed'],
            projections_path=projections_path, alpha=0.0,
            adaptive_offsets=cfg['direction']['adaptive_offsets'],
            batch_size=cfg['steering'].get('logit_batch_size', 16),
        )
        save_pkl(result, out_path)
        if (layer + 1) % 10 == 0:
            log(f"    Layer {layer} done")

    # Window sweep
    win_size = cfg['layer_sweeps']['window_size']
    n_windows = n_layers - win_size + 1
    log(f"\n  Window sweep ({n_windows} windows of {win_size} layers)...")
    for start in range(n_windows):
        end = start + win_size - 1
        layers = list(range(start, end + 1))
        out_path = os.path.join(sweep_dir, 'window', f'L{start}-{end}.pkl')
        if os.path.exists(out_path):
            continue
        result = run_logit_steering_hf(
            bundle=bundle, data=dev_data,
            direction_path=direction_path,
            layers=layers, offsets=cfg['steering']['offsets'],
            scale=0, seed=cfg['seed'],
            projections_path=projections_path, alpha=0.0,
            adaptive_offsets=cfg['direction']['adaptive_offsets'],
            batch_size=cfg['steering'].get('logit_batch_size', 16),
        )
        save_pkl(result, out_path)
        if (start + 1) % 10 == 0:
            log(f"    Window L{start}-{end} done")

    del bundle
    free_gpu()

    # Find best window
    best_delta = 0
    best_win = None
    for start in range(n_windows):
        end = start + win_size - 1
        r = load_steering_flip_rate(os.path.join(sweep_dir, 'window', f'L{start}-{end}.pkl'))
        if r and r[2] < best_delta:
            best_delta = r[2]
            best_win = (start, end)

    if best_win:
        log(f"\n  Best window: L{best_win[0]}-{best_win[1]} (ΔLD={best_delta:+.2f})")
        save_pkl({'best_window': best_win, 'best_delta': best_delta, 'n_layers': n_layers},
                 os.path.join(sweep_dir, 'best_window.pkl'))

    # Generate sweep plots
    _generate_sweep_plots(cfg, output_dir, n_layers)


# ---------------------------------------------------------------------------
# Stage 6: Logit Steering (test set)
# ---------------------------------------------------------------------------

def stage_logit_steering(cfg, output_dir):
    """Run logit steering on test set with all layers and best window."""
    log("\n" + "=" * 60)
    log("  STAGE 6: LOGIT STEERING (test)")
    log("=" * 60)

    from src.hf_utils import load_model_hf
    from src.logit_steering_hf_batched import run_logit_steering_hf
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(cfg['model_name'])
    n_layers = getattr(config, 'num_hidden_layers', None) or config.text_config.num_hidden_layers

    direction_path = os.path.join(output_dir, 'direction', 'direction.pt')
    projections_path = os.path.join(output_dir, 'direction', 'projections.pkl')
    test_data_path = os.path.join(output_dir, 'data', 'test.pkl')

    best_win_path = os.path.join(output_dir, 'layer_sweeps', 'best_window.pkl')
    best_win = load_pkl(best_win_path)['best_window'] if os.path.exists(best_win_path) else None

    ls_dir = os.path.join(output_dir, 'logit_steering')
    os.makedirs(ls_dir, exist_ok=True)

    bundle = load_model_hf(
        cfg['model_name'],
        load_in_8bit=cfg.get('load_in_8bit', False),
        load_in_4bit=cfg.get('load_in_4bit', False),
        cache_dir=cfg.get('cache_dir'),
    )

    test_data = load_pkl(test_data_path)

    configs = []
    for alpha in cfg['steering']['alphas']:
        configs.append(('all_layers', list(range(n_layers)), alpha, f'Lall_S{alpha}'))
        if best_win:
            win_layers = list(range(best_win[0], best_win[1] + 1))
            configs.append(('best_window', win_layers, alpha,
                           f'L{best_win[0]}-{best_win[1]}_S{alpha}'))

    for label, layers, alpha, suffix in configs:
        log(f"\n  {suffix}...")
        result = run_logit_steering_hf(
            bundle=bundle, data=test_data,
            direction_path=direction_path,
            layers=layers, offsets=cfg['steering']['offsets'],
            scale=0, seed=cfg['seed'],
            projections_path=projections_path, alpha=alpha,
            adaptive_offsets=cfg['direction']['adaptive_offsets'],
            batch_size=cfg['steering'].get('logit_batch_size', 16),
        )
        save_pkl(result, os.path.join(ls_dir, f'{suffix}.pkl'))

    del bundle
    free_gpu()

    # Report results
    _report_logit_steering(cfg, output_dir)


# ---------------------------------------------------------------------------
# Stage 7: CoT Steering (test set)
# ---------------------------------------------------------------------------

def stage_cot_steering(cfg, output_dir):
    """Run CoT steering on test set: baseline + best window steered."""
    log("\n" + "=" * 60)
    log("  STAGE 7: COT STEERING (test)")
    log("=" * 60)

    from src.intervened_generator import generate_with_intervention
    from run_steering import load_patching_pairs_as_steering_data, stage_inspect

    direction_path = os.path.join(output_dir, 'direction', 'direction.pt')
    projections_path = os.path.join(output_dir, 'direction', 'projections.pkl')
    test_data_path = os.path.join(output_dir, 'data', 'test.pkl')
    cot_dir = os.path.join(output_dir, 'cot_steering')
    os.makedirs(cot_dir, exist_ok=True)

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(cfg['model_name'])
    n_layers = getattr(config, 'num_hidden_layers', None) or config.text_config.num_hidden_layers

    # Layer selection priority: config cot_layers > CoT sweep best window > logit sweep best window
    cot_layers_cfg = cfg['steering'].get('cot_layers')
    if cot_layers_cfg:
        win_layers = cot_layers_cfg
        best_win = (min(win_layers), max(win_layers))
        log(f"  Using explicit CoT layers from config: L{best_win[0]}-{best_win[1]}")
    else:
        # Prefer CoT sweep results over logit sweep
        cot_sweep_path = os.path.join(output_dir, 'cot_layer_sweep', 'best_window.pkl')
        logit_sweep_path = os.path.join(output_dir, 'layer_sweeps', 'best_window.pkl')
        if os.path.exists(cot_sweep_path):
            best_win = load_pkl(cot_sweep_path)['best_window']
            log(f"  Using CoT sweep best window: L{best_win[0]}-{best_win[1]}")
        elif os.path.exists(logit_sweep_path):
            best_win = load_pkl(logit_sweep_path)['best_window']
            log(f"  Using logit sweep best window: L{best_win[0]}-{best_win[1]}")
        else:
            best_win = None
        win_layers = list(range(best_win[0], best_win[1] + 1)) if best_win else list(range(n_layers))

    cot_alphas = cfg['steering'].get('cot_alphas', [cfg['steering']['cot_alpha']])
    batch_size = cfg['steering']['cot_batch_size']
    cot_max_new_tokens = cfg['steering'].get('cot_max_new_tokens', 1000)
    log(f"  max_new_tokens: {cot_max_new_tokens}")

    # Baseline
    log("\n  Baseline (no intervention)...")
    from run_steering import stage_generate
    stage_generate(
        num_datapoints=9999, scale=0,
        layers=list(range(n_layers)),
        offsets=cfg['steering']['offsets'],
        suffix='baseline',
        direction_path=direction_path,
        model_name=cfg['model_name'],
        seed=cfg['seed'], batch_size=batch_size,
        data_path=test_data_path,
        adaptive_offsets=cfg['direction']['adaptive_offsets'],
        max_new_tokens=cot_max_new_tokens,
        output_dir=cot_dir,
    )

    # Steered (best window or all layers) at each alpha
    win_label = f"L{best_win[0]}-{best_win[1]}" if best_win else "Lall"
    for alpha in cot_alphas:
        log(f"\n  Steered ({win_label}, {alpha} std)...")
        stage_generate(
            num_datapoints=9999, scale=None,
            layers=win_layers,
            offsets=cfg['steering']['offsets'],
            suffix=f'{win_label}_S{alpha}',
            direction_path=direction_path,
            model_name=cfg['model_name'],
            seed=cfg['seed'], batch_size=batch_size,
            data_path=test_data_path,
            match_mean=True, mean_match_std_dev=alpha,
            projections_path=projections_path,
            adaptive_offsets=cfg['direction']['adaptive_offsets'],
            max_new_tokens=cot_max_new_tokens,
            output_dir=cot_dir,
        )

    # Inspect
    log("\n  Inspecting results...")
    stage_inspect(model_name=cfg['model_name'], suffix='baseline', output_dir=cot_dir)
    for alpha in cot_alphas:
        stage_inspect(model_name=cfg['model_name'], suffix=f'{win_label}_S{alpha}', output_dir=cot_dir)


# ---------------------------------------------------------------------------
# Plotting and Reporting Helpers
# ---------------------------------------------------------------------------

def _generate_bc_heatmaps(cfg, output_dir, split='test'):
    """Generate BC accuracy and correlation heatmaps."""
    log(f"\n  Generating BC heatmaps ({split} set)...")

    train_proj = load_pkl(os.path.join(output_dir, 'direction', 'projections.pkl'))
    if split == 'test':
        test_proj = load_pkl(os.path.join(output_dir, 'test_projections', 'projections.pkl'))
    else:
        test_proj = train_proj

    p = test_proj
    correct = p['correct_projections'].cpu().numpy()
    incorrect = p['incorrect_projections'].cpu().numpy()
    N = correct.shape[0]
    offsets = p['offsets']
    layers = p['layers']
    n_offsets, n_layers = len(offsets), len(layers)

    # Build per-question incorrect index
    has_variable = 'incorrect_question_idx' in p
    if has_variable:
        iq_idx = p['incorrect_question_idx']
    else:
        n_choices = len(set(p['correct_positions']))
        n_inc = n_choices - 1
        iq_idx = []
        for i in range(N):
            for j in range(n_inc):
                iq_idx.append(i)

    def _get_inc(q_idx):
        return [incorrect[j] for j, qi in enumerate(iq_idx) if qi == q_idx]

    # Train position means for BC (using index-based approach)
    tc = train_proj['correct_projections'].cpu().numpy()
    ti = train_proj['incorrect_projections'].cpu().numpy()
    TN = tc.shape[0]
    if 'incorrect_question_idx' in train_proj:
        t_iq_idx = train_proj['incorrect_question_idx']
    else:
        t_n_choices = len(set(train_proj['correct_positions']))
        t_n_inc = t_n_choices - 1
        t_iq_idx = []
        for i in range(TN):
            for j in range(t_n_inc):
                t_iq_idx.append(i)
    t_n_choices_list = train_proj.get('n_choices_per_question',
        [len(set(train_proj['correct_positions']))] * TN)

    pm = {}
    pprojs = {}
    for i in range(TN):
        cp = train_proj['correct_positions'][i]
        n_ch = t_n_choices_list[i] if isinstance(t_n_choices_list, list) else t_n_choices_list
        pprojs.setdefault(cp, []).append(tc[i])
        inc_for_q = [ti[j] for j, qi in enumerate(t_iq_idx) if qi == i]
        ip = [j for j in range(n_ch) if j != cp]
        for ji, j in enumerate(ip):
            if ji < len(inc_for_q):
                pprojs.setdefault(j, []).append(inc_for_q[ji])
    for pos in pprojs:
        pm[pos] = np.nanmean(np.stack(pprojs[pos]), axis=0)

    p_n_choices_list = p.get('n_choices_per_question',
        [len(set(p['correct_positions']))] * N)

    # Confidence: slice letter_logits to each question's real choice count
    # (variable-n_choices runs pad letter_logits with NaN up to the max)
    confidences = np.full(N, np.nan)
    for i in range(N):
        ll = p['letter_logits'][i].numpy()
        cp = p['correct_positions'][i]
        n_ch = p_n_choices_list[i] if isinstance(p_n_choices_list, list) else p_n_choices_list
        others = np.concatenate([ll[:cp], ll[cp+1:n_ch]])
        if others.size:
            confidences[i] = ll[cp] - np.nanmax(others)

    bc_corr = np.full((n_offsets, n_layers), np.nan)
    bc_accuracy = np.full((n_offsets, n_layers), np.nan)

    for o_idx in range(n_offsets):
        for l_idx in range(n_layers):
            bc_margins = np.zeros(N)
            valid = np.ones(N, dtype=bool)
            for i in range(N):
                cp = p['correct_positions'][i]
                n_ch = p_n_choices_list[i] if isinstance(p_n_choices_list, list) else p_n_choices_list
                ip = [j for j in range(n_ch) if j != cp]
                c = correct[i, o_idx, l_idx]
                inc_for_q = _get_inc(i)
                incs = [inc[o_idx, l_idx] if inc.ndim > 0 else float('nan') for inc in inc_for_q]
                if np.isnan(c) or not incs or any(np.isnan(x) for x in incs) or cp not in pm:
                    valid[i] = False
                    continue
                bc_c = c - pm[cp][o_idx, l_idx]
                bc_incs = [incs[ji] - pm[ip[ji]][o_idx, l_idx] for ji in range(len(incs)) if ip[ji] in pm]
                if not bc_incs:
                    valid[i] = False
                    continue
                bc_margins[i] = bc_c - max(bc_incs)
            nv = valid.sum()
            if nv > 10:
                conf_valid = valid & ~np.isnan(confidences)
                if conf_valid.sum() > 10:
                    bc_corr[o_idx, l_idx], _ = pearsonr(confidences[conf_valid], bc_margins[conf_valid])
                bc_accuracy[o_idx, l_idx] = (bc_margins[valid] > 0).sum() / nv

    # Plot
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    for data, fname, cbar, vmin, vmax, fmt in [
        (bc_corr.T, f'bc_correlation_{split}.pdf', 'Pearson r', 0, 0.8, '.2f'),
        (bc_accuracy.T * 100, f'bc_accuracy_{split}.pdf', 'Top-1 Accuracy (%)', 25, 100, '.0f'),
    ]:
        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(data, aspect='auto', cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
        for x in range(data.shape[1]):
            for y in range(data.shape[0]):
                val = data[y, x]
                if not np.isnan(val) and y % 2 == 0:
                    color = 'white' if val < vmin + 0.2*(vmax-vmin) or val > vmax - 0.2*(vmax-vmin) else 'black'
                    ax.text(x, y, f'{val:{fmt}}', ha='center', va='center', fontsize=5, color=color)
        xlabels = [str(o) for o in offsets]
        ax.set_xticks(range(len(offsets)))
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_xlabel('Position Offset', fontsize=12)
        ticks = [i for i, l in enumerate(layers) if l % 5 == 0]
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(layers[i]) for i in ticks], fontsize=10)
        ax.set_ylabel('Layer', fontsize=12)
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label(cbar, fontsize=10, rotation=270, labelpad=15)
        cb.ax.tick_params(labelsize=10)
        fig.savefig(os.path.join(plot_dir, fname), dpi=300, bbox_inches='tight')
        plt.close(fig)
        log(f"  Saved {fname}")

    log(f"  Peak BC accuracy: {np.nanmax(bc_accuracy)*100:.1f}%")
    log(f"  Peak BC correlation: {np.nanmax(bc_corr):.3f}")


def _generate_sweep_plots(cfg, output_dir, n_layers):
    """Generate per-layer and window sweep plots."""
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    for sweep_type, subdir, xlabel, is_window in [
        ('per_layer', 'per_layer', 'Layer', False),
        ('window', 'window', 'Window Center Layer', True),
    ]:
        xs, ys = [], []
        sweep_dir = os.path.join(output_dir, 'layer_sweeps', subdir)

        if sweep_type == 'per_layer':
            for layer in range(n_layers):
                r = load_steering_flip_rate(os.path.join(sweep_dir, f'L{layer}.pkl'))
                if r:
                    xs.append(layer)
                    ys.append(r[1] * 100)
        else:
            win_size = cfg['layer_sweeps']['window_size']
            for start in range(n_layers - win_size + 1):
                end = start + win_size - 1
                r = load_steering_flip_rate(os.path.join(sweep_dir, f'L{start}-{end}.pkl'))
                if r:
                    xs.append((start + end) / 2)
                    ys.append(r[1] * 100)

        if not xs:
            continue

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(xs, ys, linewidth=1.5, color='#1f77b4', zorder=3)
        ax.fill_between(xs, 0, ys, alpha=0.15, color='#1f77b4', zorder=2)

        peak_idx = np.argmax(ys)
        ax.plot(xs[peak_idx], ys[peak_idx], 'o', color='#1f77b4', markersize=6, zorder=4)
        if is_window:
            s = int(xs[peak_idx] - (win_size - 1) / 2)
            e = int(xs[peak_idx] + (win_size - 1) / 2)
            txt = f'L{s}-{e}: {ys[peak_idx]:.0f}%'
        else:
            txt = f'L{int(xs[peak_idx])}: {ys[peak_idx]:.0f}%'
        ax.annotate(txt, xy=(xs[peak_idx], ys[peak_idx]),
                    xytext=(12, 5), textcoords='offset points',
                    fontsize=9, color='#1f77b4', fontweight='bold',
                    arrowprops=dict(arrowstyle='-', color='#1f77b4', alpha=0.5, lw=0.8))

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Flipped (%)', fontsize=12)
        ax.set_ylim(-5, 105)
        ax.tick_params(labelsize=10)
        fig.savefig(os.path.join(plot_dir, f'{sweep_type}_sweep.pdf'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        log(f"  Saved {sweep_type}_sweep.pdf")


def _report_logit_steering(cfg, output_dir):
    """Report logit steering results, split by dataset."""
    ls_dir = os.path.join(output_dir, 'logit_steering')
    test_data = load_pkl(os.path.join(output_dir, 'data', 'test.pkl'))
    pairs = test_data['pairs']
    datasets = cfg['datasets']

    log("\n  LOGIT STEERING RESULTS (test set):")
    log(f"  {'Config':<25} {'Set':<12} {'N':>5} {'Flipped':>8} {'Rate':>7} {'ΔLD':>8}")
    log(f"  {'-'*68}")

    for fname in sorted(os.listdir(ls_dir)):
        if not fname.endswith('.pkl'):
            continue
        suffix = fname.replace('.pkl', '')
        r = load_pkl(os.path.join(ls_dir, fname))
        s = r['steered_ld'].numpy() if torch.is_tensor(r['steered_ld']) else np.array(r['steered_ld'])
        d = r['delta_ld'].numpy() if torch.is_tensor(r['delta_ld']) else np.array(r['delta_ld'])
        n = len(s)

        # Overall
        v = ~np.isnan(s)
        fl = (s[v] < 0).sum()
        log(f"  {suffix:<25} {'Overall':<12} {v.sum():>5} {fl:>8} {fl/v.sum():>6.1%} {np.nanmean(d):>+8.2f}")

        # Per dataset
        for ds in datasets:
            mask = np.array([pairs[i].get('source_dataset', '') == ds
                           for i in range(min(n, len(pairs)))]) & v[:min(n, len(pairs))]
            nv = mask.sum()
            if nv == 0:
                continue
            fl_ds = (s[:min(n, len(pairs))][mask] < 0).sum()
            log(f"  {'':<25} {ds:<12} {nv:>5} {fl_ds:>8} {fl_ds/nv:>6.1%} {np.nanmean(d[:min(n, len(pairs))][mask]):>+8.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_STAGES = ['filter', 'direction', 'patching', 'test_projections',
              'layer_sweeps', 'logit_steering', 'cot_steering']

def main():
    parser = argparse.ArgumentParser(description='Final evaluation pipeline')
    parser.add_argument('--config', required=True, help='Path to config YAML')
    parser.add_argument('--output_dir', required=True, help='Output directory for this run')
    parser.add_argument('--stages', nargs='+', default=ALL_STAGES,
                        choices=ALL_STAGES, help='Stages to run (default: all)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    import shutil
    shutil.copy2(args.config, os.path.join(args.output_dir, 'config.yaml'))

    log("=" * 60)
    log(f"  FINAL PIPELINE: {cfg['model_name']}")
    log(f"  Datasets: {cfg['datasets']}")
    log(f"  Output: {args.output_dir}")
    log(f"  Stages: {args.stages}")
    log("=" * 60)

    import time
    start = time.time()

    stage_map = {
        'filter': stage_filter_and_split,
        'direction': stage_direction,
        'patching': stage_patching,
        'test_projections': stage_test_projections,
        'layer_sweeps': stage_layer_sweeps,
        'logit_steering': stage_logit_steering,
        'cot_steering': stage_cot_steering,
    }

    for stage_name in args.stages:
        stage_fn = stage_map[stage_name]
        stage_fn(cfg, args.output_dir)

    elapsed = time.time() - start
    log(f"\n{'=' * 60}")
    log(f"  PIPELINE COMPLETE: {elapsed/60:.1f} minutes")
    log(f"{'=' * 60}")


if __name__ == '__main__':
    main()
