"""
Plotting and token-inspection helpers for random-swap patching results.

The patching itself runs in activation_patching_hf.py; this module renders its
output (heatmaps, per-layer line plots, token-length splits).

Metric (normalized logit difference):
  baseline = l1 - l2  (from unpatched Q1, positive)
  corrupt  = l1 - l2  (from unpatched Q2, negative)
  patched  = l1 - l2  (from patched Q1)
  metric   = (baseline - patched) / (baseline - corrupt)
  0 = no effect, 1 = full flip
"""

import gc
import os
import random
from typing import List, Dict, Optional

import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.adaptive_utils import (
    get_answer_token_length_at_position, clip_offsets_for_token_length,
    get_min_offsets, group_pairs_by_token_length,
    verify_token_lengths, save_debug_file,
)


def get_example_tokens_at_offsets(
    tokenizer,
    pair: Dict,
    offsets: List[int],
) -> List[str]:
    """Extract example tokens at each offset position from the first element of a pair.

    Uses the correct answer's delimiter position as the reference point.

    Args:
        tokenizer: The model's tokenizer
        pair: A single pair dict from data_generator
        offsets: Position offsets relative to delimiter

    Returns:
        List of token strings, one per offset
    """
    delim_pos = pair['delimiter_positions'][pair['correct_pos']]

    token_ids = tokenizer.encode(pair['gen'], add_special_tokens=False)

    tokens = []
    for offset in offsets:
        pos = delim_pos + offset
        if 0 <= pos < len(token_ids):
            tok_str = tokenizer.decode([token_ids[pos]])
            # Clean up for display: strip whitespace, truncate long tokens
            tok_str = tok_str.strip()
            if len(tok_str) > 15:
                tok_str = tok_str[:12] + "..."
            tokens.append(tok_str)
        else:
            tokens.append("?")

    return tokens


def plot_patching_heatmap(
    result: Dict,
    output_path: str,
    title: str = "Activation Patching: Swap Patching Heatmap",
    example_tokens: Optional[List[str]] = None,
) -> None:
    """Plot a heatmap of patching effects: layer (y) x offset (x).

    If the result has all_offsets=True (1D per-layer data), delegates to
    plot_patching_lineplot instead.

    Args:
        result: Dict from run_swap_patching
        output_path: Path to save the PNG
        title: Plot title
        example_tokens: Optional list of example token strings (one per offset)
            to show on x-axis alongside the offset number
    """
    # Check if this is all_offsets (1D) data
    if result.get('metadata', {}).get('all_offsets', False):
        plot_patching_lineplot(result, output_path, title=title)
        return

    mean_metrics = result['mean_metrics']
    layers = result['layers']
    offsets = result['offsets']

    if torch.is_tensor(mean_metrics):
        data = mean_metrics.cpu().numpy()
    else:
        data = np.array(mean_metrics)

    fig, ax = plt.subplots(figsize=(max(8, len(offsets) * 1.5), max(10, len(layers) * 0.4)))

    im = ax.imshow(
        data,
        aspect='auto',
        cmap='RdBu_r',
        origin='lower',
        vmin=0,
        vmax=max(0.3, np.nanmax(data)),
    )

    # X-axis: offset labels with optional example tokens
    ax.set_xticks(range(len(offsets)))
    if example_tokens and len(example_tokens) == len(offsets):
        xlabels = [f"{o}\n\"{tok}\"" for o, tok in zip(offsets, example_tokens)]
    else:
        xlabels = [str(o) for o in offsets]
    ax.set_xticklabels(xlabels, fontsize=9)
    if example_tokens and len(example_tokens) == len(offsets):
        ax.set_xlabel('Position Offset (relative to answer delimiter)\nExample tokens shown below offset')
    else:
        ax.set_xlabel('Position Offset (relative to answer delimiter)')

    # Show every layer on y-axis
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([str(l) for l in layers])
    ax.set_ylabel('Layer')

    ax.set_title(title)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Normalized Logit Difference\n(0=no effect, 1=full flip)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved heatmap to {output_path}")


def plot_patching_lineplot(
    result: Dict,
    output_path: str,
    title: str = "Activation Patching: All Offsets Combined",
) -> None:
    """Plot a line plot of per-layer patching effects (all offsets patched simultaneously).

    Args:
        result: Dict from run_swap_patching with all_offsets=True
        output_path: Path to save the PNG
        title: Plot title
    """
    mean_metrics = result['mean_metrics']
    layers = result['layers']

    if torch.is_tensor(mean_metrics):
        data = mean_metrics.cpu().numpy()
    else:
        data = np.array(mean_metrics)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(layers, data, marker='o', linewidth=2, markersize=4, color='#d62728')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.fill_between(layers, 0, data, alpha=0.15, color='#d62728')

    ax.set_xlabel('Layer')
    ax.set_ylabel('Normalized Logit Difference\n(0=no effect, 1=full flip)')
    ax.set_title(title)
    ax.set_xlim(layers[0], layers[-1])

    # Add offset info to subtitle
    offsets = result['offsets']
    ax.text(0.5, -0.12, f'Offsets patched simultaneously: {offsets}',
            transform=ax.transAxes, ha='center', fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved line plot to {output_path}")


def plot_patching_lineplot_by_token_length(
    result: Dict,
    tokenizer,
    output_path: str,
    title: str = "Activation Patching: All Offsets by Token Length",
) -> None:
    """Plot per-layer patching effect split by correct answer token length.

    One line per token length group, all on the same axes.
    """
    metrics = result['metrics']  # (n_pairs, n_layers)
    layers = result['layers']
    pairs = result.get('pairs')

    if pairs is None:
        print("  Warning: pairs not in result, cannot split by token length", flush=True)
        return

    groups = group_pairs_by_token_length(tokenizer, pairs)

    colors = {3: '#1f77b4', 4: '#ff7f0e', 5: '#2ca02c', 6: '#d62728', 7: '#9467bd'}
    fig, ax = plt.subplots(figsize=(10, 6))

    for tl in sorted(groups.keys()):
        indices = groups[tl]
        group_data = metrics[indices]  # (n_group, n_layers)
        mean_data = torch.nanmean(group_data, dim=0).cpu().numpy()
        color = colors.get(tl, None)
        ax.plot(layers, mean_data, marker='o', linewidth=2, markersize=4,
                label=f'{tl}-token (n={len(indices)})', color=color)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Normalized Logit Difference\n(0=no effect, 1=full flip)')
    ax.set_title(title)
    ax.set_xlim(layers[0], layers[-1])
    ax.legend()

    offsets = result['offsets']
    ax.text(0.5, -0.12, f'Offsets patched simultaneously: {offsets}',
            transform=ax.transAxes, ha='center', fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved token-length line plot to {output_path}")


def plot_patching_heatmaps_by_token_length(
    result: Dict,
    tokenizer,
    output_dir: str,
    title_prefix: str = "Patching",
) -> None:
    """Plot separate heatmaps for each correct-answer token length group.

    Groups pairs by the correct answer's token length, computes
    nanmean per group, and plots a heatmap clipped to valid offsets for each group.
    Only meaningful for per-offset results (not all_offsets mode).

    Args:
        result: Dict from run_*_patching with per-offset metrics
        tokenizer: Model tokenizer (needed if pairs not in result, but groups precomputed)
        output_dir: Directory to save heatmap PNGs
        title_prefix: Prefix for plot titles
    """
    if result.get('metadata', {}).get('all_offsets', False):
        print("  Skipping per-token-length heatmaps (all_offsets mode)", flush=True)
        return

    metrics = result['metrics']  # (n_pairs, n_layers, n_offsets)
    layers = result['layers']
    offsets = result['offsets']
    pairs = result.get('pairs')  # May not be in result

    if pairs is None:
        print("  Warning: pairs not in result dict, cannot group by token length", flush=True)
        return

    groups = group_pairs_by_token_length(tokenizer, pairs)
    os.makedirs(output_dir, exist_ok=True)

    for tl, indices in sorted(groups.items()):
        # Get metrics for this group
        group_metrics = metrics[indices]  # (n_group, n_layers, n_offsets)

        # Compute valid offsets for this token length
        valid = clip_offsets_for_token_length(offsets, tl)
        valid_indices = [offsets.index(o) for o in valid if o in offsets]

        # Slice to valid offsets
        if valid_indices:
            group_data = group_metrics[:, :, valid_indices]
        else:
            continue

        mean_data = torch.nanmean(group_data, dim=0).cpu().numpy()

        fig, ax = plt.subplots(figsize=(max(8, len(valid) * 1.5), max(10, len(layers) * 0.4)))

        im = ax.imshow(
            mean_data,
            aspect='auto',
            cmap='RdBu_r',
            origin='lower',
            vmin=0,
            vmax=max(0.3, np.nanmax(mean_data)) if not np.all(np.isnan(mean_data)) else 0.3,
        )

        ax.set_xticks(range(len(valid)))
        ax.set_xticklabels([str(o) for o in valid], fontsize=9)
        ax.set_xlabel('Position Offset')
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([str(l) for l in layers])
        ax.set_ylabel('Layer')
        ax.set_title(f"{title_prefix}: {tl}-token answers")

        plt.colorbar(im, ax=ax, label='Normalized Logit Difference')
        plt.tight_layout()

        path = os.path.join(output_dir, f"patching_heatmap_{tl}tok.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {tl}-token heatmap ({len(indices)} pairs) to {path}", flush=True)
