"""Library function for creating bias-corrected projections."""

import pickle
import numpy as np
import torch


def _get_incorrect_for_question(incorrect, incorrect_question_idx, q_idx):
    """Get incorrect projections for a specific question."""
    if incorrect_question_idx is not None:
        mask = [i for i, qi in enumerate(incorrect_question_idx) if qi == q_idx]
        return [incorrect[i] for i in mask]
    else:
        # Legacy format: assume uniform n_choices, grouped sequentially
        return None


def create_bc_projections(input_path, output_path):
    """Create bias-corrected projections from existing projections file.

    Handles both fixed n_choices (legacy reshape) and variable n_choices
    (using incorrect_question_idx).
    """
    with open(input_path, 'rb') as f:
        p = pickle.load(f)

    correct = p['correct_projections'].cpu().numpy()
    incorrect = p['incorrect_projections'].cpu().numpy()
    N = correct.shape[0]

    # Check if we have variable n_choices metadata
    has_variable = 'incorrect_question_idx' in p and 'n_choices_per_question' in p
    n_choices_list = p.get('n_choices_per_question', None)

    if has_variable:
        iq_idx = p['incorrect_question_idx']
        all_n_choices = set(n_choices_list)
        max_n_choices = max(all_n_choices)
    else:
        # Legacy: derive n_choices from shapes
        n_choices = len(set(p['correct_positions'])) if len(p['correct_positions']) > 10 else max(p['correct_positions']) + 1
        n_inc = n_choices - 1
        if incorrect.shape[0] != N * n_inc:
            n_inc = incorrect.shape[0] // N
            n_choices = n_inc + 1
        max_n_choices = n_choices
        n_choices_list = [n_choices] * N
        # Build question index from reshape
        iq_idx = []
        for i in range(N):
            for j in range(n_inc):
                iq_idx.append(i)

    # Build position projections for BC means
    position_projs = {}
    for i in range(N):
        cp = p['correct_positions'][i]
        n_ch = n_choices_list[i]

        if cp not in position_projs:
            position_projs[cp] = []
        position_projs[cp].append(correct[i])

        # Get this question's incorrect projections
        inc_for_q = [incorrect[j] for j, qi in enumerate(iq_idx) if qi == i]
        inc_positions = [j for j in range(n_ch) if j != cp]
        for ji, j in enumerate(inc_positions):
            if ji < len(inc_for_q):
                if j not in position_projs:
                    position_projs[j] = []
                position_projs[j].append(inc_for_q[ji])

    position_means = {}
    for pos in position_projs:
        if position_projs[pos]:
            position_means[pos] = np.nanmean(np.stack(position_projs[pos]), axis=0)

    # Compute BC projections
    bc_correct = np.zeros_like(correct)
    bc_incorrect_list = []
    for i in range(N):
        cp = p['correct_positions'][i]
        bc_correct[i] = correct[i] - position_means[cp]

        n_ch = n_choices_list[i]
        inc_for_q = [incorrect[j] for j, qi in enumerate(iq_idx) if qi == i]
        inc_positions = [j for j in range(n_ch) if j != cp]
        for ji, j in enumerate(inc_positions):
            if ji < len(inc_for_q):
                bc_incorrect_list.append(inc_for_q[ji] - position_means[j])

    bc_incorrect_flat = np.stack(bc_incorrect_list) if bc_incorrect_list else np.zeros((0,) + correct.shape[1:])

    bc_proj = {
        'correct_projections': torch.tensor(bc_correct, dtype=torch.float32),
        'incorrect_projections': torch.tensor(bc_incorrect_flat, dtype=torch.float32),
        'correct_mean': torch.tensor(np.nanmean(bc_correct, axis=0), dtype=torch.float32),
        'incorrect_mean': torch.tensor(np.nanmean(bc_incorrect_flat, axis=0), dtype=torch.float32),
        'correct_std': torch.tensor(np.nanstd(bc_correct, axis=0), dtype=torch.float32),
        'incorrect_std': torch.tensor(np.nanstd(bc_incorrect_flat, axis=0), dtype=torch.float32),
        'separation': torch.tensor(np.nanmean(bc_correct, axis=0) - np.nanmean(bc_incorrect_flat, axis=0), dtype=torch.float32),
        'offsets': p['offsets'],
        'layers': p['layers'],
        'position_means': {pos: torch.tensor(position_means[pos], dtype=torch.float32) for pos in position_means},
        'bias_corrected': True,
    }

    if has_variable:
        bc_proj['n_choices_per_question'] = n_choices_list
        bc_proj['incorrect_question_idx'] = iq_idx
        bc_proj['variable_n_choices'] = True

    for key in ['letter_logits', 'correct_positions', 'questions', 'answer_choices']:
        if key in p:
            bc_proj[key] = p[key]

    with open(output_path, 'wb') as f:
        pickle.dump(bc_proj, f)
    print(f"Saved BC projections to {output_path}", flush=True)
