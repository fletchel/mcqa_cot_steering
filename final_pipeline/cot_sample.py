#!/usr/bin/env python3
"""Run a CoT steering sample on a subset of an existing run's test set.

The direction/projections and the evaluation data can come from different
runs (cross-direction steering). Baseline generation is optional — the data
run's existing cot_steering baseline already covers its test questions.

Usage:
    python final_pipeline/cot_sample.py --data_run runs/qwen_big \
        --source_dataset hellaswag --num_datapoints 200 --run_baseline
    python final_pipeline/cot_sample.py \
        --data_run /path/to/qwen25_7b_hellaswag_5k \
        --direction_run runs/qwen_mmlu_5k --num_datapoints 300
"""

import argparse
import os
import pickle
import random
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

import yaml
from run_steering import stage_generate, stage_inspect


def main():
    p = argparse.ArgumentParser(description='CoT steering sample')
    p.add_argument('--data_run', required=True, help='run dir providing data/test.pkl')
    p.add_argument('--direction_run', default=None,
                   help='run dir providing direction/projections (default: data_run)')
    p.add_argument('--source_dataset', default=None,
                   help='restrict to pairs from this dataset (default: all)')
    p.add_argument('--num_datapoints', type=int, default=300)
    p.add_argument('--subset_seed', type=int, default=42)
    p.add_argument('--alpha', type=float, default=2.0)
    p.add_argument('--layers', default='all',
                   help="'all' or comma-separated layer indices, e.g. 17,18,19,20,21")
    p.add_argument('--out_subdir', default='cot_sample')
    p.add_argument('--run_baseline', action='store_true',
                   help='also generate a no-intervention baseline')
    p.add_argument('--batch_size', type=int, default=24)
    p.add_argument('--max_new_tokens', type=int, default=1000)
    args = p.parse_args()

    data_run = args.data_run
    direction_run = args.direction_run or data_run

    with open(os.path.join(data_run, 'config.yaml')) as f:
        cfg = yaml.safe_load(f)
    model_name = cfg['model_name']

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name)
    n_layers = getattr(config, 'num_hidden_layers', None) or config.text_config.num_hidden_layers

    if args.layers == 'all':
        layers = list(range(n_layers))
        win_label = 'Lall'
    else:
        layers = [int(x) for x in args.layers.split(',')]
        win_label = f'L{min(layers)}-{max(layers)}'

    out_dir = os.path.join(data_run, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # Build the subset data file
    with open(os.path.join(data_run, 'data', 'test.pkl'), 'rb') as f:
        test = pickle.load(f)
    pairs = test['pairs']
    if args.source_dataset:
        pairs = [p_ for p_ in pairs if p_.get('source_dataset') == args.source_dataset]
    print(f"{len(pairs)} candidate pairs", flush=True)
    if len(pairs) > args.num_datapoints:
        pairs = random.Random(args.subset_seed).sample(pairs, args.num_datapoints)
    subset_path = os.path.join(out_dir, 'data_subset.pkl')
    with open(subset_path, 'wb') as f:
        pickle.dump({'pairs': pairs,
                     'metadata': {**test.get('metadata', {}), 'n': len(pairs),
                                  'subset_of': 'test',
                                  'source_dataset': args.source_dataset}}, f)
    print(f"Saved {len(pairs)} pairs to {subset_path}", flush=True)

    direction_path = os.path.join(direction_run, 'direction', 'direction.pt')
    projections_path = os.path.join(direction_run, 'direction', 'projections.pkl')

    # Tag cross-direction outputs with the direction source
    dir_tag = '' if direction_run == data_run else f"dir-{os.path.basename(direction_run.rstrip('/'))}_"
    suffix = f"{dir_tag}{win_label}_S{args.alpha}"

    common = dict(
        num_datapoints=9999,
        offsets=cfg['direction']['offsets'],
        direction_path=direction_path,
        model_name=model_name,
        seed=cfg['seed'],
        batch_size=args.batch_size,
        data_path=subset_path,
        adaptive_offsets=cfg['direction']['adaptive_offsets'],
        max_new_tokens=args.max_new_tokens,
        output_dir=out_dir,
    )

    if args.run_baseline:
        print("\nBaseline...", flush=True)
        stage_generate(scale=0, layers=list(range(n_layers)),
                       suffix=f'{dir_tag}baseline', **common)

    print(f"\nSteered ({suffix})...", flush=True)
    stage_generate(scale=None, layers=layers, suffix=suffix,
                   match_mean=True, mean_match_std_dev=args.alpha,
                   projections_path=projections_path, **common)

    print("\nInspecting...", flush=True)
    if args.run_baseline:
        stage_inspect(model_name=model_name, suffix=f'{dir_tag}baseline', output_dir=out_dir)
    stage_inspect(model_name=model_name, suffix=suffix, output_dir=out_dir)


if __name__ == '__main__':
    main()
