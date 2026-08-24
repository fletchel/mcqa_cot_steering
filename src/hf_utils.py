"""HuggingFace utilities replacing TransformerLens for the correctness direction pipeline.

Provides:
  - ModelBundle: thin wrapper around HF model + tokenizer + config
  - load_model_hf(): load model with optional quantization and multi-GPU
  - run_with_cache_hf(): forward pass capturing residual stream activations
  - run_with_hooks_hf(): forward pass with activation-modifying hooks
  - get_letter_token_ids_hf(): get token IDs for [a, b, c, d]

All functions operate on raw HuggingFace models (AutoModelForCausalLM) and
use register_forward_hook for activation capture/modification.
"""

import gc
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# The cuDNN SDPA backend fails on some node/driver combos ("cudnn_frontend ...
# No valid execution plans built", seen with Qwen3-32B on H100). Disable it so
# torch falls back to its flash/mem-efficient SDPA kernels.
try:
    torch.backends.cuda.enable_cudnn_sdp(False)
except AttributeError:
    pass


# ---------------------------------------------------------------------------
# Model layer access (architecture-agnostic)
# ---------------------------------------------------------------------------
def _get_model_layers(model):
    """Return the list of transformer layer modules.

    Supports Qwen, Llama, Mistral (model.model.layers),
    GPT-2/GPT-J (model.transformer.h), and others.
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
    raise ValueError(
        f"Cannot find transformer layers for {type(model).__name__}. "
        "Expected model.model.layers, model.model.language_model.layers, "
        "model.transformer.h, or model.transformer.blocks"
    )


def _get_layer_count(model) -> int:
    """Return total number of transformer layers."""
    return len(_get_model_layers(model))


def _get_d_model(model) -> int:
    """Return hidden dimension (d_model) from model config."""
    cfg = model.config
    for attr in ('hidden_size', 'd_model', 'n_embd'):
        if hasattr(cfg, attr):
            return getattr(cfg, attr)
    # Gemma 3 conditional generation: hidden_size is in text_config
    if hasattr(cfg, 'text_config') and hasattr(cfg.text_config, 'hidden_size'):
        return cfg.text_config.hidden_size
    raise ValueError(f"Cannot determine d_model from config: {cfg}")


# ---------------------------------------------------------------------------
# ModelBundle: thin wrapper for pipeline convenience
# ---------------------------------------------------------------------------
@dataclass
class ModelBundle:
    """Container holding an HF model + tokenizer + derived config.

    Provides attribute access matching the TransformerLens patterns used
    throughout the pipeline (n_layers, d_model, device, dtype, tokenizer).
    """
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    n_layers: int = 0
    d_model: int = 0
    device: torch.device = field(default_factory=lambda: torch.device('cpu'))
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        if self.n_layers == 0:
            self.n_layers = _get_layer_count(self.model)
        if self.d_model == 0:
            self.d_model = _get_d_model(self.model)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_pretrained_causal_lm(model_name: str, **kwargs):
    """Load with AutoModelForCausalLM, falling back to AutoModelForImageTextToText
    for multimodal wrappers whose config lacks top-level LM fields (e.g. Qwen3.6,
    which registers only as Qwen3_5ForConditionalGeneration)."""
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except (ValueError, AttributeError, KeyError):
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)


def load_model_hf(
    model_name: str,
    dtype: torch.dtype = torch.bfloat16,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    device_map: str = 'auto',
    cache_dir: Optional[str] = None,
    attn_implementation: Optional[str] = None,
) -> ModelBundle:
    """Load an HF causal LM with optional quantization and multi-GPU support.

    Args:
        model_name: HuggingFace model ID or local path
        dtype: Torch dtype (ignored if quantized)
        load_in_8bit: 8-bit quantization via bitsandbytes
        load_in_4bit: 4-bit quantization via bitsandbytes
        device_map: Device placement strategy ('auto' for multi-GPU)
        cache_dir: HF cache directory (e.g. /scratch-shared/...)
        attn_implementation: Attention implementation (e.g. 'flash_attention_2')

    Returns:
        ModelBundle with model, tokenizer, and derived config
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    kwargs = {
        'device_map': device_map,
        'cache_dir': cache_dir,
    }

    if attn_implementation:
        kwargs['attn_implementation'] = attn_implementation

    if load_in_4bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type='nf4',
        )
    elif load_in_8bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs['torch_dtype'] = dtype

    model = load_pretrained_causal_lm(model_name, **kwargs)
    model.eval()

    # Determine effective device (may be 'meta' for multi-GPU, use first param)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')

    print(f"Loaded {model_name}: {_get_layer_count(model)} layers, "
          f"{_get_d_model(model)} dims, device={device}", flush=True)

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------
def tokenize_text(bundle: ModelBundle, text: str, return_tensors: bool = True):
    """Tokenize text and move to model device.

    Returns:
        If return_tensors: dict with input_ids, attention_mask on device
        Else: list of token IDs
    """
    if return_tensors:
        encoded = bundle.tokenizer(text, return_tensors='pt', add_special_tokens=False)
        return {k: v.to(bundle.device) for k, v in encoded.items()}
    return bundle.tokenizer.encode(text, add_special_tokens=False)


def get_letter_token_ids_hf(tokenizer, device=None, n_choices: int = 4) -> torch.Tensor:
    """Get token IDs for answer letters.

    Returns tensor of shape (n_choices,) on the specified device.
    """
    import string
    letters = list(string.ascii_lowercase[:n_choices])
    ids = tokenizer.convert_tokens_to_ids(letters)
    t = torch.tensor(ids)
    if device is not None:
        t = t.to(device)
    return t


# ---------------------------------------------------------------------------
# run_with_cache_hf: forward pass with residual stream capture
# ---------------------------------------------------------------------------
def run_with_cache_hf(
    bundle: ModelBundle,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    layers: Optional[List[int]] = None,
    hook_point: str = 'output',
) -> tuple:
    """Forward pass capturing residual stream activations at specified layers.

    Equivalent to TransformerLens model.run_with_cache(text, names_filter=...).

    Hooks are placed on each transformer layer module. By default captures
    the layer output (post-MLP residual stream, equivalent to hook_resid_post).

    Args:
        bundle: ModelBundle with model and config
        input_ids: Token IDs, shape (batch, seq_len) or (seq_len,)
        attention_mask: Optional attention mask, shape matching input_ids
        layers: Layer indices to capture (None = all layers)
        hook_point: 'output' for post-layer residual (hook_resid_post equivalent)

    Returns:
        (logits, cache) where:
          - logits: shape (batch, seq_len, vocab_size)
          - cache: dict mapping layer_idx -> tensor of shape (batch, seq_len, d_model)
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask is not None and attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    if layers is None:
        layers = list(range(bundle.n_layers))

    model_layers = _get_model_layers(bundle.model)
    cache = {}
    handles = []

    for layer_idx in layers:
        def make_hook(li):
            def hook_fn(module, input, output):
                # Most HF models return (hidden_states, ...) from each layer
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                cache[li] = hidden.detach().float().cpu()
            return hook_fn

        h = model_layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        handles.append(h)

    try:
        with torch.no_grad():
            outputs = bundle.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        logits = outputs.logits
    finally:
        for h in handles:
            h.remove()

    return logits, cache


def run_forward_hf(
    bundle: ModelBundle,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Simple forward pass returning logits only.

    Equivalent to TransformerLens model(text, return_type='logits').
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask is not None and attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    with torch.no_grad():
        outputs = bundle.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    return outputs.logits


# ---------------------------------------------------------------------------
# run_with_hooks_hf: forward pass with activation-modifying hooks
# ---------------------------------------------------------------------------
def run_with_hooks_hf(
    bundle: ModelBundle,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    fwd_hooks: Optional[List[tuple]] = None,
) -> torch.Tensor:
    """Forward pass with activation-modifying hooks. Returns logits.

    Equivalent to TransformerLens model.run_with_hooks(text, fwd_hooks=...).

    Args:
        bundle: ModelBundle
        input_ids: Token IDs, shape (batch, seq_len) or (seq_len,)
        attention_mask: Optional attention mask
        fwd_hooks: List of (layer_idx, hook_fn) tuples. Each hook_fn receives
            (module, input, output) and should return modified output.
            The hook_fn signature matches HF register_forward_hook:
              hook_fn(module, input, output) -> modified_output or None

            For convenience, if hook_fn takes 2 args (activation, layer_idx),
            it will be wrapped automatically to extract the hidden states
            from the output tuple, apply the function, and repack.

    Returns:
        logits: shape (batch, seq_len, vocab_size)
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask is not None and attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    model_layers = _get_model_layers(bundle.model)
    handles = []

    if fwd_hooks:
        for layer_idx, hook_fn in fwd_hooks:
            # Wrap hook to handle HF layer output format: (hidden_states, ...)
            def make_wrapper(fn):
                def wrapper(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                        modified = fn(hidden)
                        if modified is not None:
                            return (modified,) + output[1:]
                        return output
                    else:
                        result = fn(output)
                        return result if result is not None else output
                return wrapper

            h = model_layers[layer_idx].register_forward_hook(make_wrapper(hook_fn))
            handles.append(h)

    try:
        with torch.no_grad():
            outputs = bundle.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        logits = outputs.logits
    finally:
        for h in handles:
            h.remove()

    return logits


# ---------------------------------------------------------------------------
# Batched tokenization for patching
# ---------------------------------------------------------------------------
def tokenize_and_pad(
    tokenizer,
    texts: List[str],
    device: torch.device,
) -> Dict:
    """Tokenize a list of texts, pad to max length, return tensors.

    Returns dict with 'input_ids', 'attention_mask', 'lengths'.
    """
    token_ids_list = [
        tokenizer.encode(t, add_special_tokens=False)
        for t in texts
    ]
    lengths = [len(t) for t in token_ids_list]
    max_len = max(lengths)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    B = len(texts)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)

    for b in range(B):
        input_ids[b, :lengths[b]] = torch.tensor(token_ids_list[b])
        attention_mask[b, :lengths[b]] = 1

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'lengths': lengths,
    }
