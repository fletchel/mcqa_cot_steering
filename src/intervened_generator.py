"""Generation with correctness direction intervention using native PyTorch hooks.

Model-agnostic version: works with any HuggingFace causal LM (Qwen, Llama, Gemma, etc.).

Generates MMLU responses while intervening on residual streams at answer
delimiter positions to flip model confidence between correct and incorrect answers.

The intervention:
- Subtracts scale * direction from the correct answer's delimiter positions
- Adds scale * direction to a random incorrect answer's delimiter positions

This is designed to make the model more likely to answer with the target (incorrect)
answer by manipulating its internal representation of "correctness" at those positions.
"""

import gc
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple
from contextlib import contextmanager

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import get_prompt_answer_positions
from src.hf_utils import load_pretrained_causal_lm
from src.adaptive_utils import (
    get_answer_token_length, clip_offsets_for_token_length,
    verify_token_lengths,
)

# Additional imports for MMLU loading
from datasets import load_dataset
from torch.utils.data import DataLoader


def get_model_layers(model):
    """Return the list of transformer layers for any supported architecture.

    Tries common layer-list attribute paths:
      - model.model.layers   (Llama, Qwen, Gemma, Mistral)
      - model.transformer.h  (GPT-2, GPT-J, GPT-NeoX)
      - model.transformer.blocks  (MPT)

    Raises a clear error if none match.
    """
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    # Gemma 3 (conditional generation): model.model.language_model.layers
    if hasattr(model, 'model') and hasattr(model.model, 'language_model') and hasattr(model.model.language_model, 'layers'):
        return model.model.language_model.layers
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'blocks'):
        return model.transformer.blocks
    raise AttributeError(
        f"Cannot find transformer layers for {type(model).__name__}. "
        f"Expected one of: model.model.layers, model.transformer.h, model.transformer.blocks. "
        f"Top-level attributes: {[a for a in dir(model) if not a.startswith('_')]}"
    )


def _format_dataset_item(d: dict, dataset: str = 'mmlu') -> Tuple[str, int, List[str]]:
    """Format a dataset example as a multiple choice question with letter labels."""
    if dataset == 'hellaswag':
        choices = list(d['endings'])
        correct_idx = int(d['label'])
        question = d['ctx']
    else:
        choices = list(d['choices'])
        correct_idx = d['answer']
        question = d['question']

    import string
    letters = list(string.ascii_lowercase[:len(choices)])
    labeled_choices = [f"({letters[i]}) {choices[i]}" for i in range(len(choices))]
    formatted_q = question + '\nChoices: ' + ', '.join(labeled_choices) + '.\n'
    return formatted_q, correct_idx, choices


# Keep old name as alias for backward compatibility
def format_mmlu(d: dict) -> Tuple[str, int, List[str]]:
    return _format_dataset_item(d, dataset='mmlu')


def get_dataset_dataloader(
    tokenizer,
    num_datapoints: int,
    batch_size: int,
    seed: int = 42,
    dataset: str = 'mmlu',
) -> DataLoader:
    """Create a DataLoader for the specified dataset."""
    from src.data_generator import DATASET_INSTRUCTIONS

    random.seed(seed)

    if dataset == 'hellaswag':
        ds = load_dataset('Rowan/hellaswag')['validation']
        choices_key = 'endings'
    else:
        ds = load_dataset('cais/mmlu', 'all')['test']
        choices_key = 'choices'

    # Filter: exclude items where any answer choice has >5 tokens or contains a comma
    print(f"Filtering {dataset} dataset...", flush=True)
    filtered_indices = []
    for idx in range(len(ds)):
        choices = ds[idx][choices_key]
        valid = True
        for choice in choices:
            if ',' in choice:
                valid = False
                break
            tokens = tokenizer.encode(choice, add_special_tokens=False)
            if len(tokens) > 5:
                valid = False
                break
        if valid:
            filtered_indices.append(idx)

    print(f"Filtered {len(ds)} -> {len(filtered_indices)} examples", flush=True)

    if len(filtered_indices) < num_datapoints:
        raise ValueError(f"Only {len(filtered_indices)} examples pass filters, but {num_datapoints} requested")

    random_indices = random.sample(filtered_indices, k=num_datapoints)
    ds = [ds[i] for i in random_indices]

    instruction = DATASET_INSTRUCTIONS.get(dataset, DATASET_INSTRUCTIONS['mmlu'])
    question_prepend = f'{instruction} Think step-by-step before giving your answer. \n'
    chat_questions = []

    for i in range(num_datapoints):
        formatted_question, correct_answer, answer_texts = _format_dataset_item(ds[i], dataset=dataset)
        question_format = [{"role": "user", "content": question_prepend + formatted_question}]
        chat_question = tokenizer.apply_chat_template(
            question_format,
            tokenize=False,
            add_generation_prompt=True
        )
        chat_questions.append((chat_question, correct_answer, answer_texts))

    return DataLoader(chat_questions, batch_size=batch_size)


# Keep old name as alias
def get_mmlu_dataloader(tokenizer, num_datapoints, batch_size, seed=42):
    return get_dataset_dataloader(tokenizer, num_datapoints, batch_size, seed, dataset='mmlu')


class InterventionHook:
    """Hook that modifies residual stream at specific positions.

    Supports batched interventions where each batch item has different positions.
    """

    def __init__(
        self,
        correct_positions_batch: List[List[int]],
        target_positions_batch: List[List[int]],
        direction_vecs: List[torch.Tensor],
        scale: float,
        normal_noise: bool = False,
        noise_scale: float = 100.0,
        d_model: int = None,
        match_mean: bool = False,
        correct_proj_targets: List[float] = None,
        target_proj_targets: List[float] = None,
        layer_idx: int = None,
        debug_store: dict = None,
        allowed_offset_ids: set = None,
    ):
        """
        Args:
            correct_positions_batch: List of positions per batch item, each is list of positions per offset
            target_positions_batch: List of positions per batch item, each is list of positions per offset
            direction_vecs: List of direction vectors (one per position offset)
            scale: Intervention strength (ignored in match_mean mode)
            normal_noise: If True, add random noise to ALL positions instead of direction intervention
            noise_scale: Scale of random noise when normal_noise=True
            d_model: Model hidden dimension (for noise generation). Required if normal_noise=True,
                     otherwise inferred from direction_vecs.
            match_mean: If True, steer projections to opposite class mean instead of fixed scale
            correct_proj_targets: Target projections for correct positions (one per offset, match_mean only)
            target_proj_targets: Target projections for target positions (one per offset, match_mean only)
            layer_idx: Layer index (for debug logging)
            debug_store: Shared dict to store debug info; keys are (batch_idx, layer_idx, offset_idx)
        """
        self.correct_positions_batch = correct_positions_batch
        self.target_positions_batch = target_positions_batch
        self.direction_vecs = direction_vecs
        self.scale = scale
        self.normal_noise = normal_noise
        self.noise_scale = noise_scale
        self.match_mean = match_mean
        self.correct_proj_targets = correct_proj_targets
        self.target_proj_targets = target_proj_targets
        self.layer_idx = layer_idx
        self.debug_store = debug_store
        # Restrict which offsets this layer's hook applies (indices into the
        # per-offset lists). None = all offsets (default behavior).
        self.allowed_offset_ids = allowed_offset_ids
        if d_model is not None:
            self.d_model = d_model
        elif direction_vecs:
            self.d_model = direction_vecs[0].shape[-1]
        else:
            raise ValueError("d_model must be provided when direction_vecs is empty")

    def __call__(self, module, input, output):
        """Hook function called during forward pass."""
        # Handle both tuple and tensor output formats
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # During generation with KV-cache, after the first forward pass (full prompt),
        # subsequent passes only process one token at a time, giving shape [batch, hidden_dim].
        # We only want to intervene during the initial prompt processing (3D tensor).
        if hidden_states.dim() != 3:
            return output

        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        # Clone to avoid in-place modification issues
        hidden_states = hidden_states.clone()

        if self.normal_noise:
            # TEST MODE: Add large random noise to ALL positions
            noise = torch.randn(
                hidden_states.shape,
                device=hidden_states.device,
                dtype=hidden_states.dtype
            ) * self.noise_scale
            hidden_states = hidden_states + noise
        else:
            # Normal intervention: apply direction at specific positions per batch item
            for batch_idx in range(min(batch_size, len(self.correct_positions_batch))):
                correct_positions = self.correct_positions_batch[batch_idx]
                target_positions = self.target_positions_batch[batch_idx]

                for offset_idx, (correct_pos, target_pos) in enumerate(
                    zip(correct_positions, target_positions)
                ):
                    if self.allowed_offset_ids is not None and offset_idx not in self.allowed_offset_ids:
                        continue
                    dir_vec = self.direction_vecs[offset_idx].to(hidden_states.device)

                    if self.match_mean:
                        # Match-mean: adjust projection to opposite class mean
                        c_tgt = self.correct_proj_targets[offset_idx]
                        t_tgt = self.target_proj_targets[offset_idx]
                        correct_proj_val = None
                        target_proj_val = None

                        if correct_pos < seq_len:
                            correct_proj_val = (hidden_states[batch_idx, correct_pos, :] * dir_vec).sum()
                            hidden_states[batch_idx, correct_pos, :] = (
                                hidden_states[batch_idx, correct_pos, :] + (c_tgt - correct_proj_val) * dir_vec
                            )

                        if target_pos < seq_len:
                            target_proj_val = (hidden_states[batch_idx, target_pos, :] * dir_vec).sum()
                            hidden_states[batch_idx, target_pos, :] = (
                                hidden_states[batch_idx, target_pos, :] + (t_tgt - target_proj_val) * dir_vec
                            )

                        # Only store debug info when positions are in range
                        # (skip subsequent KV-cache passes where seq_len=1)
                        if self.debug_store is not None and (correct_proj_val is not None or target_proj_val is not None):
                            correct_proj_after = None
                            target_proj_after = None
                            if correct_pos < seq_len:
                                correct_proj_after = float((hidden_states[batch_idx, correct_pos, :] * dir_vec).sum())
                            if target_pos < seq_len:
                                target_proj_after = float((hidden_states[batch_idx, target_pos, :] * dir_vec).sum())
                            self.debug_store[(batch_idx, self.layer_idx, offset_idx)] = {
                                'correct_proj': float(correct_proj_val) if correct_proj_val is not None else None,
                                'target_proj': float(target_proj_val) if target_proj_val is not None else None,
                                'correct_target': float(c_tgt),
                                'target_target': float(t_tgt),
                                'correct_proj_after': correct_proj_after,
                                'target_proj_after': target_proj_after,
                            }
                    else:
                        # Fixed scale: add/subtract scale * direction
                        if correct_pos < seq_len:
                            hidden_states[batch_idx, correct_pos, :] = hidden_states[batch_idx, correct_pos, :] - self.scale * dir_vec

                        if target_pos < seq_len:
                            hidden_states[batch_idx, target_pos, :] = hidden_states[batch_idx, target_pos, :] + self.scale * dir_vec

        # Return modified output (keep other elements unchanged)
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        else:
            return hidden_states


@contextmanager
def intervention_hooks(
    model,
    correct_positions_batch: List[List[int]],
    target_positions_batch: List[List[int]],
    direction: torch.Tensor,
    intervention_layers: List[int],
    offset_indices: List[int],
    scale: float,
    device: torch.device,
    dtype: torch.dtype,
    normal_noise: bool = False,
    noise_scale: float = 100.0,
    d_model: int = None,
    match_mean: bool = False,
    mean_match_std_dev: float = 0.0,
    proj_means: dict = None,
    debug_store: dict = None,
    cell_mask: dict = None,
):
    """Context manager that registers intervention hooks on model layers.

    Args:
        correct_positions_batch: List of positions per batch item
        target_positions_batch: List of positions per batch item
        match_mean: If True, steer projections to opposite class mean
        mean_match_std_dev: Std dev multiplier for match_mean mode
            (correct → incorrect_mean - n*std, target → correct_mean + n*std)
        proj_means: Dict mapping (offset_list_index, layer) to
            (correct_mean, incorrect_mean, correct_std, incorrect_std)
        debug_store: Shared dict for match-mean debug logging (populated by hooks)
        cell_mask: Optional {layer_idx: set of offset-list indices}. When given,
            only listed layers get hooks and each applies only its listed offsets.
    """
    handles = []
    layers = get_model_layers(model)

    try:
        for layer_idx in intervention_layers:
            if cell_mask is not None and layer_idx not in cell_mask:
                continue
            # Get direction vectors for this layer at each offset
            direction_vecs = []
            correct_proj_targets = []
            target_proj_targets = []
            for oi, offset_idx in enumerate(offset_indices):
                vec = direction[offset_idx, layer_idx].to(device).to(dtype)
                direction_vecs.append(vec)

                if match_mean and proj_means is not None:
                    correct_mean, incorrect_mean, correct_std, incorrect_std = proj_means[(oi, layer_idx)]
                    # correct pos → incorrect mean - n*std (push further down)
                    # target pos → correct mean + n*std (push further up)
                    correct_proj_targets.append(incorrect_mean - mean_match_std_dev * incorrect_std)
                    target_proj_targets.append(correct_mean + mean_match_std_dev * correct_std)

            # Create hook
            hook = InterventionHook(
                correct_positions_batch=correct_positions_batch,
                target_positions_batch=target_positions_batch,
                direction_vecs=direction_vecs,
                scale=scale,
                normal_noise=normal_noise,
                noise_scale=noise_scale,
                d_model=d_model,
                match_mean=match_mean,
                correct_proj_targets=correct_proj_targets if match_mean else None,
                target_proj_targets=target_proj_targets if match_mean else None,
                layer_idx=layer_idx,
                debug_store=debug_store,
                allowed_offset_ids=cell_mask[layer_idx] if cell_mask is not None else None,
            )

            # Register hook on the layer (works for any architecture)
            layer = layers[layer_idx]
            handle = layer.register_forward_hook(hook)
            handles.append(handle)

        yield

    finally:
        # Remove all hooks
        for handle in handles:
            handle.remove()


def generate_single_with_intervention(
    model,
    tokenizer,
    prompt: str,
    correct_idx: int,
    answer_texts: List[str],
    direction: torch.Tensor,
    intervention_layers: List[int],
    intervention_offsets: List[int],
    offset_indices: List[int],
    scale: float,
    max_new_tokens: int = 500,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.8,
    seed: int = 42,
    device: torch.device = None,
    dtype: torch.dtype = None,
    normal_noise: bool = False,
    noise_scale: float = 100.0,
    match_mean: bool = False,
    mean_match_std_dev: float = 0.0,
    proj_means: dict = None,
) -> Dict:
    """
    Generate a response with correctness direction intervention using native hooks.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: The chat-templated prompt
        correct_idx: Index of the correct answer (0-3)
        answer_texts: List of 4 answer texts
        direction: Direction tensor of shape (n_offsets, n_layers, d_model)
        intervention_layers: Which layers to intervene at
        intervention_offsets: Which position offsets to use (e.g., [-1, 0, 1])
        offset_indices: Indices into the direction tensor for each offset
        scale: Intervention strength
        max_new_tokens: Max tokens to generate
        do_sample: Use sampling vs greedy
        temperature: Sampling temperature
        top_p: Top-p sampling
        seed: Random seed for target selection
        device: Torch device
        dtype: Torch dtype

    Returns:
        Dict with generation results and intervention metadata
    """
    random.seed(seed)

    # Pick a random incorrect answer as target
    incorrect_indices = [i for i in range(len(answer_texts)) if i != correct_idx]
    target_idx = random.choice(incorrect_indices)

    # Find delimiter positions
    pos_info = get_prompt_answer_positions(tokenizer, prompt, answer_texts)

    if not pos_info['success']:
        return {
            'success': False,
            'error': 'Could not find all answer positions',
            'positions': pos_info['positions'],
            'correct_idx': correct_idx,
            'target_idx': target_idx,
        }

    delimiter_positions = pos_info['positions']

    # Compute intervention positions for correct and target answers
    correct_positions = [delimiter_positions[correct_idx] + off for off in intervention_offsets]
    target_positions = [delimiter_positions[target_idx] + off for off in intervention_offsets]

    # Tokenize prompt (no special tokens — positions from get_prompt_answer_positions
    # use add_special_tokens=False, so input_ids must match)
    input_ids = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False).to(device)
    prompt_len = input_ids.shape[1]

    # Prepare generation kwargs
    gen_kwargs = {
        'max_new_tokens': max_new_tokens,
        'do_sample': do_sample,
        'pad_token_id': tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs['temperature'] = temperature
        gen_kwargs['top_p'] = top_p

    # Generate with intervention hooks
    with intervention_hooks(
        model=model,
        correct_positions_batch=[correct_positions],  # Wrap in list for batch interface
        target_positions_batch=[target_positions],
        direction=direction,
        intervention_layers=intervention_layers,
        offset_indices=offset_indices,
        scale=scale,
        device=device,
        dtype=dtype,
        normal_noise=normal_noise,
        noise_scale=noise_scale,
        d_model=direction.shape[-1],
        match_mean=match_mean,
        mean_match_std_dev=mean_match_std_dev,
        proj_means=proj_means,
    ):
        with torch.no_grad():
            output_ids = model.generate(input_ids, **gen_kwargs)

    # Decode response
    output_ids_list = output_ids[0].tolist()
    response_tokens = output_ids_list[prompt_len:]
    response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

    full_generation = tokenizer.decode(output_ids_list, skip_special_tokens=False)

    return {
        'success': True,
        'prompt': prompt,
        'response': response_text,
        'generation': full_generation,
        'correct_idx': correct_idx,
        'target_idx': target_idx,
        'answer_texts': answer_texts,
        'delimiter_positions': delimiter_positions,
        'correct_positions': correct_positions,
        'target_positions': target_positions,
        'intervention_layers': intervention_layers,
        'intervention_offsets': intervention_offsets,
        'scale': scale,
    }


def generate_batch_with_intervention(
    model,
    tokenizer,
    batch_data: List[Tuple[str, int, List[str]]],  # List of (prompt, correct_idx, answer_texts)
    direction: torch.Tensor,
    intervention_layers: List[int],
    intervention_offsets: List[int],
    offset_indices: List[int],
    scale: float,
    max_new_tokens: int = 500,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.8,
    seed: int = 42,
    device: torch.device = None,
    dtype: torch.dtype = None,
    normal_noise: bool = False,
    noise_scale: float = 100.0,
    match_mean: bool = False,
    mean_match_std_dev: float = 0.0,
    proj_means: dict = None,
    debug_store: dict = None,
    adaptive_offsets: bool = False,
    cell_mask: dict = None,
) -> List[Dict]:
    """
    Generate responses for a batch with correctness direction intervention.

    Uses left-padding for batched generation and adjusts positions accordingly.
    """
    random.seed(seed)
    batch_size = len(batch_data)
    results = []

    # Prepare batch: find positions, pick targets, tokenize
    prompts = []
    correct_indices = []
    target_indices = []
    answer_texts_list = []
    delimiter_positions_list = []
    original_lengths = []

    for i, (prompt, correct_idx, answer_texts) in enumerate(batch_data):
        prompts.append(prompt)
        correct_indices.append(correct_idx)
        answer_texts_list.append(answer_texts)

        # Pick random incorrect answer as target
        random.seed(seed + i)
        incorrect = [j for j in range(len(answer_texts)) if j != correct_idx]
        target_idx = random.choice(incorrect)
        target_indices.append(target_idx)

        # Find delimiter positions
        pos_info = get_prompt_answer_positions(tokenizer, prompt, answer_texts)
        if not pos_info['success']:
            delimiter_positions_list.append(None)
        else:
            delimiter_positions_list.append(pos_info['positions'])

        # Get original token length
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        original_lengths.append(len(tokens))

    # Tokenize with left padding for generation
    tokenizer.padding_side = 'left'
    encoded = tokenizer(
        prompts,
        return_tensors='pt',
        padding=True,
        add_special_tokens=False,
    ).to(device)

    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    max_len = input_ids.shape[1]

    # Compute padded positions for each batch item
    correct_positions_batch = []
    target_positions_batch = []

    # Sentinel value: any position >= seq_len will be skipped by InterventionHook's
    # `if correct_pos < seq_len` guards.
    SENTINEL = max_len + 1000

    for i in range(batch_size):
        if delimiter_positions_list[i] is None:
            # Skip this item - use dummy positions that won't be in range
            correct_positions_batch.append([-1] * len(intervention_offsets))
            target_positions_batch.append([-1] * len(intervention_offsets))
            continue

        # Calculate padding offset (left padding)
        pad_offset = max_len - original_lengths[i]
        delimiter_positions = delimiter_positions_list[i]

        if adaptive_offsets:
            # Compute valid offsets based on answer token lengths
            answer_texts = batch_data[i][2]  # (prompt, correct_idx, answer_texts)
            correct_tl = get_answer_token_length(tokenizer, answer_texts[correct_indices[i]])
            target_tl = get_answer_token_length(tokenizer, answer_texts[target_indices[i]])
            min_tl = min(correct_tl, target_tl)
            valid_offsets = set(clip_offsets_for_token_length(intervention_offsets, min_tl))

            correct_positions = []
            target_positions = []
            for off in intervention_offsets:
                if off in valid_offsets:
                    correct_positions.append(delimiter_positions[correct_indices[i]] + off + pad_offset)
                    target_positions.append(delimiter_positions[target_indices[i]] + off + pad_offset)
                else:
                    correct_positions.append(SENTINEL)
                    target_positions.append(SENTINEL)
        else:
            # Standard: all offsets for all positions
            correct_positions = [delimiter_positions[correct_indices[i]] + off + pad_offset for off in intervention_offsets]
            target_positions = [delimiter_positions[target_indices[i]] + off + pad_offset for off in intervention_offsets]

        correct_positions_batch.append(correct_positions)
        target_positions_batch.append(target_positions)

    # Prepare generation kwargs
    gen_kwargs = {
        'max_new_tokens': max_new_tokens,
        'do_sample': do_sample,
        'pad_token_id': tokenizer.pad_token_id or tokenizer.eos_token_id,
        'attention_mask': attention_mask,
    }
    if do_sample:
        gen_kwargs['temperature'] = temperature
        gen_kwargs['top_p'] = top_p

    # Generate with intervention hooks
    with intervention_hooks(
        model=model,
        correct_positions_batch=correct_positions_batch,
        target_positions_batch=target_positions_batch,
        direction=direction,
        intervention_layers=intervention_layers,
        offset_indices=offset_indices,
        scale=scale,
        device=device,
        dtype=dtype,
        normal_noise=normal_noise,
        noise_scale=noise_scale,
        d_model=direction.shape[-1],
        match_mean=match_mean,
        mean_match_std_dev=mean_match_std_dev,
        proj_means=proj_means,
        debug_store=debug_store,
        cell_mask=cell_mask,
    ):
        with torch.no_grad():
            output_ids = model.generate(input_ids, **gen_kwargs)

    # Decode each response
    for i in range(batch_size):
        if delimiter_positions_list[i] is None:
            results.append({
                'success': False,
                'error': 'Could not find all answer positions',
                'correct_idx': correct_indices[i],
                'target_idx': target_indices[i],
            })
            continue

        # Get this item's output, accounting for padding
        pad_offset = max_len - original_lengths[i]
        prompt_len = original_lengths[i]

        output_tokens = output_ids[i].tolist()
        # Skip padding tokens at the start
        output_tokens = output_tokens[pad_offset:]
        response_tokens = output_tokens[prompt_len:]

        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        full_generation = tokenizer.decode(output_tokens, skip_special_tokens=False)

        # Un-adjust positions for reporting (back to original coordinates)
        correct_positions = [p - pad_offset for p in correct_positions_batch[i]]
        target_positions = [p - pad_offset for p in target_positions_batch[i]]

        results.append({
            'success': True,
            'prompt': prompts[i],
            'response': response_text,
            'generation': full_generation,
            'correct_idx': correct_indices[i],
            'target_idx': target_indices[i],
            'answer_texts': answer_texts_list[i],
            'delimiter_positions': delimiter_positions_list[i],
            'correct_positions': correct_positions,
            'target_positions': target_positions,
            'intervention_layers': intervention_layers,
            'intervention_offsets': intervention_offsets,
            'scale': scale,
        })

    return results


def _print_match_mean_debug(debug_store, offsets, layers, batch_idx):
    """Print match-mean debug tables for a single batch item."""
    off_headers = [f"Off {o:+d}" for o in offsets]
    col_w = 16
    header = "       " + "".join(f"{h:^{col_w}}" for h in off_headers)
    sub    = "       " + "".join(f"{'corr / targ':^{col_w}}" for _ in offsets)
    sep    = "       " + "-" * (col_w * len(offsets))

    def fmt_pair(a, b):
        a_s = f"{a:+.2f}" if a is not None else " N/A "
        b_s = f"{b:+.2f}" if b is not None else " N/A "
        return f"{a_s}/{b_s}"

    print(f"\n  Actual projections:")
    print(header)
    print(sub)
    print(sep)
    for li in layers:
        row = f"  L{li:3d}: "
        for oi in range(len(offsets)):
            key = (batch_idx, li, oi)
            if key in debug_store:
                d = debug_store[key]
                cell = fmt_pair(d['correct_proj'], d['target_proj'])
            else:
                cell = "    N/A     "
            row += f"{cell:^{col_w}}"
        print(row)

    print(f"\n  Mean targets:")
    print(header)
    print(sub)
    print(sep)
    for li in layers:
        row = f"  L{li:3d}: "
        for oi in range(len(offsets)):
            key = (batch_idx, li, oi)
            if key in debug_store:
                d = debug_store[key]
                cell = fmt_pair(d['correct_target'], d['target_target'])
            else:
                cell = "    N/A     "
            row += f"{cell:^{col_w}}"
        print(row)

    print(f"\n  Steering delta (target - actual):")
    print(header)
    print(sub)
    print(sep)
    for li in layers:
        row = f"  L{li:3d}: "
        for oi in range(len(offsets)):
            key = (batch_idx, li, oi)
            if key in debug_store:
                d = debug_store[key]
                cd = (d['correct_target'] - d['correct_proj']) if d['correct_proj'] is not None else None
                td = (d['target_target'] - d['target_proj']) if d['target_proj'] is not None else None
                cell = fmt_pair(cd, td)
            else:
                cell = "    N/A     "
            row += f"{cell:^{col_w}}"
        print(row)

    print(f"\n  Post-intervention projections:")
    print(header)
    print(sub)
    print(sep)
    for li in layers:
        row = f"  L{li:3d}: "
        for oi in range(len(offsets)):
            key = (batch_idx, li, oi)
            if key in debug_store:
                d = debug_store[key]
                cell = fmt_pair(d['correct_proj_after'], d['target_proj_after'])
            else:
                cell = "    N/A     "
            row += f"{cell:^{col_w}}"
        print(row)


def generate_with_intervention(
    model_name: str,
    direction_path: str,
    output_path: str,
    num_datapoints: int,
    intervention_layers: List[int],
    intervention_offsets: List[int],
    scale: float = 1.0,
    batch_size: int = 1,
    max_new_tokens: int = 500,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.8,
    seed: int = 42,
    cache_dir: str = None,
    load_in_8bit: bool = True,
    load_in_4bit: bool = False,
    fp32: bool = False,
    verbose: bool = True,
    debug: bool = False,
    normal_noise: bool = False,
    noise_scale: float = 100.0,
    preloaded_data: List[Tuple[str, int, List[str]]] = None,
    match_mean: bool = False,
    mean_match_std_dev: float = 0.0,
    projections_path: str = None,
    mean_matching_debug: bool = False,
    adaptive_offsets: bool = False,
    dataset: str = 'mmlu',
    cell_mask: dict = None,
) -> Dict:
    """
    Generate MCQ responses with correctness direction intervention.

    This is the main entry point for intervened generation.

    Args:
        model_name: HuggingFace model name
        direction_path: Path to saved direction tensor (.pt file)
        output_path: Path to save output pickle
        num_datapoints: Number of examples to generate
        intervention_layers: Which layers to intervene at (list of layer indices)
        intervention_offsets: Position offsets relative to delimiter (e.g., [-1, 0, 1])
        scale: Intervention strength (how much of direction to add/subtract)
        batch_size: Batch size (only 1 currently supported)
        max_new_tokens: Maximum new tokens per generation
        do_sample: Use sampling vs greedy decoding
        temperature: Sampling temperature
        top_p: Top-p sampling
        seed: Random seed
        cache_dir: HuggingFace cache directory (default: $HF_HUB_CACHE or None)
        load_in_8bit: Load model in 8-bit quantization
        verbose: Print progress
        preloaded_data: Optional list of (prompt, correct_idx, answer_texts) tuples.
            When provided, uses this data instead of loading from MMLU.
            Overrides num_datapoints.

    Returns:
        Dict with results and metadata
    """
    random.seed(seed)

    if cache_dir is None:
        cache_dir = os.environ.get('HF_HUB_CACHE', None)

    # Load direction tensor
    print(f"Loading direction from: {direction_path}", flush=True)
    direction_data = torch.load(direction_path, weights_only=False)
    direction = direction_data['directions']  # Shape: (n_offsets, n_layers, d_model)
    direction_offsets = direction_data['offsets']  # e.g., [-2, -1, 0, 1, 2]

    print(f"Direction shape: {direction.shape}")
    print(f"Direction offsets available: {direction_offsets}")
    print(f"Using intervention offsets: {intervention_offsets}")
    print(f"Using intervention layers: {intervention_layers}")
    if normal_noise:
        print(f"*** NORMAL NOISE MODE: Adding random noise (scale={noise_scale}) to ALL positions ***")

    # Map intervention_offsets to indices in the direction tensor
    offset_indices = []
    for off in intervention_offsets:
        if off in direction_offsets:
            offset_indices.append(direction_offsets.index(off))
        else:
            raise ValueError(f"Offset {off} not available in direction tensor. Available: {direction_offsets}")

    print(f"Offset indices in direction tensor: {offset_indices}")

    # Convert cell_mask from offset VALUES ({layer: [-1, 0]}) to offset-list
    # indices ({layer: {oi}}) matching the per-offset lists built above.
    cell_mask_ids = None
    if cell_mask is not None:
        cell_mask_ids = {}
        for layer, offs in cell_mask.items():
            ids = {intervention_offsets.index(o) for o in offs if o in intervention_offsets}
            if ids:
                cell_mask_ids[int(layer)] = ids
        intervention_layers = sorted(cell_mask_ids)
        n_cells = sum(len(v) for v in cell_mask_ids.values())
        print(f"Cell mask active: {n_cells} (layer, offset) cells across {len(cell_mask_ids)} layers")

    # Load projection means (and optionally std devs) if in match_mean mode
    proj_means = None
    if match_mean:
        if projections_path is None:
            raise ValueError("projections_path is required for match_mean mode")
        print(f"Loading projection means from: {projections_path}", flush=True)
        with open(projections_path, 'rb') as f:
            proj_data = pickle.load(f)
        correct_mean_proj = proj_data['correct_mean']
        incorrect_mean_proj = proj_data['incorrect_mean']
        correct_std_proj = proj_data.get('correct_std')
        incorrect_std_proj = proj_data.get('incorrect_std')
        p_offsets = list(proj_data['offsets'])
        p_layers = list(proj_data['layers'])
        if torch.is_tensor(correct_mean_proj):
            correct_mean_proj = correct_mean_proj.cpu()
            incorrect_mean_proj = incorrect_mean_proj.cpu()
        if correct_std_proj is not None and torch.is_tensor(correct_std_proj):
            correct_std_proj = correct_std_proj.cpu()
            incorrect_std_proj = incorrect_std_proj.cpu()
        proj_means = {}
        for oi, off in enumerate(intervention_offsets):
            pi = p_offsets.index(off)
            for layer in intervention_layers:
                li = p_layers.index(layer)
                correct_std = float(correct_std_proj[pi, li]) if correct_std_proj is not None else 0.0
                incorrect_std = float(incorrect_std_proj[pi, li]) if incorrect_std_proj is not None else 0.0
                proj_means[(oi, layer)] = (
                    float(correct_mean_proj[pi, li]),
                    float(incorrect_mean_proj[pi, li]),
                    correct_std,
                    incorrect_std,
                )
        print(f"Match-mean mode: loaded projection means for {len(proj_means)} (offset, layer) pairs")
        if mean_match_std_dev != 0.0:
            print(f"  Using std dev multiplier: {mean_match_std_dev}")

    # Load model and tokenizer
    print(f"\nLoading model: {model_name}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    model_kwargs = {
        'cache_dir': cache_dir,
        'device_map': 'auto',
    }
    if fp32:
        model_kwargs['torch_dtype'] = torch.float32
    elif load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        model_kwargs['load_in_8bit'] = True
    else:
        model_kwargs['torch_dtype'] = torch.bfloat16

    model = load_pretrained_causal_lm(model_name, **model_kwargs)
    model.eval()
    print(f"Model loaded!", flush=True)

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get device and dtype from model
    device = torch.device('cuda')
    if fp32:
        dtype = torch.float32
    elif load_in_4bit:
        dtype = torch.bfloat16
    elif load_in_8bit:
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    # Move direction to device
    direction = direction.to(device).to(dtype)

    # Create dataloader with actual batch size
    print(f"Batch size: {batch_size}", flush=True)
    if preloaded_data is not None:
        if num_datapoints < len(preloaded_data):
            random.seed(seed)
            preloaded_data = random.sample(preloaded_data, num_datapoints)
            print(f"Using preloaded data: {len(preloaded_data)}/{num_datapoints} examples (subsampled)", flush=True)
        else:
            num_datapoints = len(preloaded_data)
            print(f"Using preloaded data: {len(preloaded_data)} examples", flush=True)
        # Custom collate to handle variable-length answer lists
        def _variable_collate(batch):
            """Collate tuples without stacking variable-length lists."""
            n_fields = len(batch[0])
            result = []
            for field_idx in range(n_fields):
                fields = [item[field_idx] for item in batch]
                if isinstance(fields[0], str):
                    result.append(fields)  # list of strings
                elif isinstance(fields[0], int):
                    result.append(torch.tensor(fields))
                elif isinstance(fields[0], (list, tuple)):
                    result.append(fields)  # keep as list of lists
                elif isinstance(fields[0], dict):
                    result.append(fields)  # keep as list of dicts
                else:
                    result.append(fields)
            return tuple(result)
        dataloader = DataLoader(preloaded_data, batch_size=batch_size,
                                collate_fn=_variable_collate)
    else:
        print(f"\nLoading {dataset} dataset...", flush=True)
        dataloader = get_dataset_dataloader(
            tokenizer=tokenizer,
            num_datapoints=num_datapoints,
            batch_size=batch_size,
            seed=seed,
            dataset=dataset,
        )

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Generate with intervention
    results = []
    skipped = 0
    global_idx = 0

    num_batches = (num_datapoints + batch_size - 1) // batch_size
    iterator = tqdm(dataloader, desc="Generating with intervention", total=num_batches) if verbose else dataloader

    for batch_num, batch in enumerate(iterator):
        # Handle both 3-tuple (MMLU) and 4-tuple (patching pairs with debug_info) formats
        if len(batch) == 4:
            prompts, answers, answer_texts_batch, debug_info_batch = batch
        else:
            prompts, answers, answer_texts_batch = batch
            debug_info_batch = None
        current_batch_size = len(prompts)

        # Prepare batch data
        batch_data = []
        batch_debug_info = []
        for i in range(current_batch_size):
            prompt = prompts[i]
            correct_idx = answers[i].item() if hasattr(answers[i], 'item') else answers[i]

            # Handle answer_texts from DataLoader
            # With custom collate: list of lists (one per item)
            # With default collate: transposed (grouped by position)
            if isinstance(answer_texts_batch[0], (list, tuple)):
                # Custom collate: answer_texts_batch[i] = list of choices for item i
                answer_texts = list(answer_texts_batch[i])
            else:
                # Default collate: transposed — answer_texts_batch[j] = j-th choice across all items
                n_at = len(answer_texts_batch)
                if hasattr(answer_texts_batch[0], '__iter__') and not isinstance(answer_texts_batch[0], str):
                    answer_texts = [answer_texts_batch[j][i] for j in range(n_at)]
                else:
                    answer_texts = [answer_texts_batch[j][i] if i < len(answer_texts_batch[j]) else answer_texts_batch[j][0] for j in range(n_at)]

            batch_data.append((prompt, correct_idx, answer_texts))

            # Extract debug info if available
            if debug_info_batch is not None:
                # Handle transposed debug_info from DataLoader
                # DataLoader collates dicts by stacking values, so we need to extract the i-th element
                if isinstance(debug_info_batch, dict):
                    # Collated dict - extract i-th element from each value
                    debug_info = {}
                    for k, v in debug_info_batch.items():
                        if hasattr(v, 'tolist'):  # tensor
                            v_list = v.tolist()
                            debug_info[k] = v_list[i] if i < len(v_list) else v_list
                        elif isinstance(v, (list, tuple)) and len(v) > i:
                            debug_info[k] = v[i]
                        else:
                            debug_info[k] = v
                elif hasattr(debug_info_batch[0], 'keys'):
                    # List of dicts (batch_size=1 or pre-batched)
                    debug_info = debug_info_batch[i] if i < len(debug_info_batch) else debug_info_batch[0]
                else:
                    debug_info = None
                batch_debug_info.append(debug_info)
            else:
                batch_debug_info.append(None)

        try:
            # Create fresh debug store for this batch
            debug_store = {} if (mean_matching_debug and match_mean) else None

            batch_results = generate_batch_with_intervention(
                model=model,
                tokenizer=tokenizer,
                batch_data=batch_data,
                direction=direction,
                intervention_layers=intervention_layers,
                intervention_offsets=intervention_offsets,
                offset_indices=offset_indices,
                scale=scale,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                seed=seed + batch_num * batch_size,
                device=device,
                dtype=dtype,
                normal_noise=normal_noise,
                noise_scale=noise_scale,
                match_mean=match_mean,
                mean_match_std_dev=mean_match_std_dev,
                proj_means=proj_means,
                debug_store=debug_store,
                adaptive_offsets=adaptive_offsets,
                cell_mask=cell_mask_ids,
            )

            for i, result in enumerate(batch_results):
                result['batch_idx'] = global_idx
                results.append(result)

                if not result['success']:
                    skipped += 1
                    if verbose:
                        tqdm.write(f"Skipped {global_idx}: {result.get('error', 'unknown error')}")
                else:
                    # Stash debug info for end-of-run printing
                    if debug_store:
                        result['_debug_store'] = dict(debug_store)
                        result['_debug_batch_idx'] = i

                    # Stash patching pair debug info if available
                    if i < len(batch_debug_info) and batch_debug_info[i] is not None:
                        result['_pair_debug_info'] = batch_debug_info[i]
                        # Promote question/subject to top-level for easy analysis
                        if 'question' in batch_debug_info[i]:
                            result['question'] = batch_debug_info[i]['question']
                        if 'subject' in batch_debug_info[i]:
                            result['subject'] = batch_debug_info[i]['subject']

                global_idx += 1

            # Clean up after each batch to prevent memory leaks
            del batch_results, batch_data, batch_debug_info
            if debug_store is not None:
                debug_store.clear()
            gc.collect()
            torch.cuda.empty_cache()

            # Progress update
            print(f"Batch {batch_num + 1}/{num_batches} complete ({global_idx} examples processed)", flush=True)

        except Exception as e:
            import traceback
            error_tb = traceback.format_exc()
            if verbose:
                tqdm.write(f"Error in batch {batch_num}: {e}\n\n{error_tb}")
            # Mark all items in batch as failed
            for i in range(current_batch_size):
                skipped += 1
                results.append({
                    'success': False,
                    'error': str(e),
                    'traceback': error_tb,
                    'batch_idx': global_idx,
                })
                global_idx += 1

            # Clean up even on error
            gc.collect()
            torch.cuda.empty_cache()

    # Compile output
    successful = [r for r in results if r.get('success', False)]

    save_data = {
        'results': results,
        'successful_results': successful,
        'metadata': {
            'model_name': model_name,
            'direction_path': direction_path,
            'num_datapoints': num_datapoints,
            'num_successful': len(successful),
            'num_skipped': skipped,
            'intervention_layers': intervention_layers,
            'intervention_offsets': intervention_offsets,
            'scale': scale,
            'seed': seed,
            'max_new_tokens': max_new_tokens,
            'do_sample': do_sample,
            'temperature': temperature,
            'top_p': top_p,
        }
    }

    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)

    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"Total: {len(results)}, Successful: {len(successful)}, Skipped: {skipped}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")

    # Release model VRAM back to the driver so the next load in this process
    # doesn't see full GPUs and offload to CPU (device_map='auto' sizing)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return save_data
