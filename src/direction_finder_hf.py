"""HuggingFace version of direction_finder.py: extract correctness direction using HF models.

Replaces TransformerLens model loading and run_with_cache with HF equivalents.
The generate stage (CPU-only) is identical — this only replaces the GPU extract stage.

Uses src.hf_utils for model loading and cached forward passes.
"""

import gc
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm.auto import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
from src.utils import get_prompt_answer_positions

from src.hf_utils import (
    ModelBundle, load_model_hf,
    run_with_cache_hf, run_forward_hf,
    get_letter_token_ids_hf,
)

# Reuse non-model functions from existing direction_finder
from src.direction_finder import (
    model_slug,
    strip_trailing_special_tokens,
    strip_cot_hint,
    _convert_patching_pairs,
    _filter_negated,
    plot_projection_separation,
)


def add_confidence_and_filter_hf(
    bundle: ModelBundle,
    data: list,
    min_confidence: float = 3.0,
) -> tuple:
    """Compute confidence and filter — HF version of direction_finder.add_confidence_and_filter."""
    letter_ids = get_letter_token_ids_hf(bundle.tokenizer, bundle.device)
    filtered = []
    skipped = 0

    for i, dp in enumerate(data):
        gen = strip_cot_hint(dp['pair'][0]['gen'])
        prompt_idx = dp['pair'][0]['prompt_pos']

        input_ids = torch.tensor(
            bundle.tokenizer.encode(gen, add_special_tokens=False),
            device=bundle.device,
        ).unsqueeze(0)

        logits = run_forward_hf(bundle, input_ids)
        final_logits = logits[0, -1, letter_ids]

        other_logits = torch.cat([final_logits[:prompt_idx], final_logits[prompt_idx + 1:]])
        confidence = final_logits[prompt_idx] - other_logits.max()

        if min_confidence > 0 and confidence.item() < min_confidence:
            skipped += 1
            continue

        dp['pair'][0]['confidence'] = confidence
        dp['pair'][0]['letter_logits'] = final_logits.float().cpu()
        filtered.append(dp)

        if (i + 1) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    return filtered, skipped


