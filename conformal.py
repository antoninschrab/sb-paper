import os
import time
import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

@partial(jax.jit, static_argnames=['n_1', 'agg_func', 'alpha'])
def tb_precompute(calib_scores, n_1, alpha, agg_func='min'):
    n, K = calib_scores.shape
    n_2 = n - n_1
    S_ref = calib_scores[:n_1]
    S_agg = calib_scores[n_1:]

    S_ref_sorted = jnp.sort(S_ref, axis=0)

    def compute_base_rank(val, col):
        return n_1 - jnp.searchsorted(col, val, side='left')

    ranks_agg = jax.vmap(jax.vmap(compute_base_rank, in_axes=(0, None)), in_axes=(1, 1))(S_agg, S_ref_sorted).T
    P_agg = (1.0 + ranks_agg) / (n_1 + 1.0)

    if agg_func == 'min': M_agg = jnp.min(P_agg, axis=1)
    elif agg_func == 'avg': M_agg = jnp.mean(P_agg, axis=1)
    elif agg_func == 'median': M_agg = jnp.median(P_agg, axis=1)

    idx = jnp.floor(alpha * (n_2 + 1.0)).astype(jnp.int32)
    idx = jnp.maximum(0, idx)
    u_tb = jnp.sort(M_agg)[idx]

    return u_tb, S_ref_sorted

@partial(jax.jit, static_argnames=['n_1', 'agg_func'])
def tb_evaluate_candidate(S_ref_sorted, u_tb, test_scores, n_1, agg_func='min'):
    def compute_test_rank(val, col):
        return n_1 - jnp.searchsorted(col, val, side='left')

    rank_eval = jax.vmap(compute_test_rank)(test_scores, S_ref_sorted.T)
    P_eval = (1.0 + rank_eval) / (n_1 + 1.0)

    if agg_func == 'min': M_eval = jnp.min(P_eval)
    elif agg_func == 'avg': M_eval = jnp.mean(P_eval)
    elif agg_func == 'median': M_eval = jnp.median(P_eval)

    return M_eval >= u_tb

@partial(jax.jit, static_argnames=['n_1', 'agg_func', 'alpha'])
def tb_exact_prediction_set(calib_scores, preds_test, Y_true, n_1, alpha, agg_func='min'):
    u_tb, S_ref_sorted = tb_precompute(calib_scores, n_1, alpha, agg_func)
    
    test_scores_true = jnp.abs(Y_true - preds_test)
    coverage = tb_evaluate_candidate(S_ref_sorted, u_tb, test_scores_true, n_1, agg_func)
    
    K = calib_scores.shape[1]
    
    # Breakpoints and their corresponding dimensions
    Y_up = preds_test[None, :] + S_ref_sorted
    Y_down = preds_test[None, :] - S_ref_sorted
    
    y_up = Y_up.flatten()
    y_down = Y_down.flatten()
    k_idx = jnp.tile(jnp.arange(K), n_1)
    
    # +1 test rank when sweeping past lower bound, -1 when sweeping past upper bound
    y_events = jnp.concatenate([y_down, y_up])
    k_events = jnp.concatenate([k_idx, k_idx])
    test_deltas = jnp.concatenate([jnp.ones(n_1 * K, dtype=jnp.int32), -jnp.ones(n_1 * K, dtype=jnp.int32)])
    
    sort_idx = jnp.argsort(y_events)
    y_events = y_events[sort_idx]
    k_events = k_events[sort_idx]
    test_deltas = test_deltas[sort_idx]
    
    # Initial state at y -> -infinity
    counts_test_init = jnp.zeros(K, dtype=jnp.int32)
    P_test_init = (1.0 + counts_test_init) / (n_1 + 1.0)
    
    if agg_func == 'min': M_test_init = jnp.min(P_test_init)
    elif agg_func == 'avg': M_test_init = jnp.mean(P_test_init)
    elif agg_func == 'median': M_test_init = jnp.median(P_test_init)
    
    valid_init = M_test_init >= u_tb
    init_state = (counts_test_init, y_events[0], valid_init)
    
    # jax.lax.scan sweeps breakpoints incrementally in linear sequential time
    def scan_step(state, event_idx):
        counts_test, y_prev, valid_prev = state
        
        y_curr = y_events[event_idx]
        k = k_events[event_idx]
        td = test_deltas[event_idx]
        
        width_contrib = jnp.where(valid_prev, y_curr - y_prev, 0.0)
        
        # state updates
        counts_test = counts_test.at[k].add(td)
        P_test = (1.0 + counts_test) / (n_1 + 1.0)
        
        if agg_func == 'min': M_test = jnp.min(P_test)
        elif agg_func == 'avg': M_test = jnp.mean(P_test)
        elif agg_func == 'median': M_test = jnp.median(P_test)
        
        valid_curr = M_test >= u_tb
        
        return (counts_test, y_curr, valid_curr), width_contrib

    _, width_contribs = jax.lax.scan(scan_step, init_state, jnp.arange(2 * n_1 * K))
    total_width = jnp.sum(width_contribs)
    
    return jnp.empty(0), jnp.empty(0), total_width, coverage

@partial(jax.jit)
def sb_precompute(calib_scores):
    n, K = calib_scores.shape
    S_calib_sorted = jnp.sort(calib_scores, axis=0)
    
    def compute_base_rank(val, col):
        return n - jnp.searchsorted(col, val, side='left')
        
    ranks_calib = jax.vmap(jax.vmap(compute_base_rank, in_axes=(0, None)), in_axes=(1, 1))(calib_scores, S_calib_sorted).T
    return S_calib_sorted, ranks_calib

@partial(jax.jit, static_argnames=['agg_func', 'alpha'])
def sb_evaluate_candidate(S_calib_sorted, ranks_calib, test_scores, n, alpha, agg_func='min'):
    n_plus_1 = n + 1.0
    
    def compute_test_rank(val, col):
        return n - jnp.searchsorted(col, val, side='left')
        
    rank_test = jax.vmap(compute_test_rank)(test_scores, S_calib_sorted.T)
    P_test = (1.0 + rank_test) / n_plus_1
    
    shift_mask = test_scores[None, :] >= S_calib_sorted
    updated_ranks_calib = ranks_calib + shift_mask.astype(jnp.int32)
    P_calib = (1.0 + updated_ranks_calib) / n_plus_1
    
    if agg_func == 'min': 
        M_test = jnp.min(P_test)
        M_calib = jnp.min(P_calib, axis=1)
    elif agg_func == 'avg': 
        M_test = jnp.mean(P_test)
        M_calib = jnp.mean(P_calib, axis=1)
    elif agg_func == 'median': 
        M_test = jnp.median(P_test)
        M_calib = jnp.median(P_calib, axis=1)
        
    final_test_rank = jnp.sum(M_calib <= M_test)
    p_sb = (1.0 + final_test_rank) / n_plus_1
    return p_sb > alpha

@partial(jax.jit, static_argnames=['agg_func', 'alpha'])
def sb_exact_prediction_set(calib_scores, preds_test, Y_true, alpha, agg_func='min'):
    n, K = calib_scores.shape
    S_calib_sorted, ranks_calib = sb_precompute(calib_scores)
    
    test_scores_true = jnp.abs(Y_true - preds_test)
    coverage = sb_evaluate_candidate(S_calib_sorted, ranks_calib, test_scores_true, n, alpha, agg_func)
    
    # Breakpoints and their corresponding calibration rows and dimensions
    Y_up = preds_test[None, :] + calib_scores
    Y_down = preds_test[None, :] - calib_scores
    
    y_up = Y_up.flatten()
    y_down = Y_down.flatten()
    
    j_idx = jnp.repeat(jnp.arange(n), K)
    k_idx = jnp.tile(jnp.arange(K), n)
    
    y_events = jnp.concatenate([y_down, y_up])
    k_events = jnp.concatenate([k_idx, k_idx])
    j_events = jnp.concatenate([j_idx, j_idx])
    
    test_deltas = jnp.concatenate([jnp.ones(n * K, dtype=jnp.int32), -jnp.ones(n * K, dtype=jnp.int32)])
    calib_deltas = jnp.concatenate([-jnp.ones(n * K, dtype=jnp.int32), jnp.ones(n * K, dtype=jnp.int32)])
    
    sort_idx = jnp.argsort(y_events)
    y_events = y_events[sort_idx]
    k_events = k_events[sort_idx]
    j_events = j_events[sort_idx]
    test_deltas = test_deltas[sort_idx]
    calib_deltas = calib_deltas[sort_idx]
    
    # Initial state at y -> -infinity
    counts_test_init = jnp.zeros(K, dtype=jnp.int32)
    counts_calib_init = ranks_calib + 1
    
    P_test_init = (1.0 + counts_test_init) / (n + 1.0)
    P_calib_init = (1.0 + counts_calib_init) / (n + 1.0)
    
    if agg_func == 'min':
        M_test_init = jnp.min(P_test_init)
        M_calib_init = jnp.min(P_calib_init, axis=1)
    elif agg_func == 'avg':
        M_test_init = jnp.mean(P_test_init)
        M_calib_init = jnp.mean(P_calib_init, axis=1)
    elif agg_func == 'median':
        M_test_init = jnp.median(P_test_init)
        M_calib_init = jnp.median(P_calib_init, axis=1)
        
    rank_test_init = jnp.sum(M_calib_init <= M_test_init)
    valid_init = (1.0 + rank_test_init) / (n + 1.0) > alpha
    
    init_state = (counts_test_init, counts_calib_init, M_calib_init, y_events[0], valid_init)
    
    # jax.lax.scan sweeps breakpoints incrementally in linear sequential time
    def scan_step(state, event_idx):
        counts_test, counts_calib, M_calib, y_prev, valid_prev = state
        
        y_curr = y_events[event_idx]
        k = k_events[event_idx]
        j = j_events[event_idx]
        td = test_deltas[event_idx]
        cd = calib_deltas[event_idx]
        
        width_contrib = jnp.where(valid_prev, y_curr - y_prev, 0.0)
        
        # Increment exactly ONE calibration rank and ONE test rank
        counts_test = counts_test.at[k].add(td)
        counts_calib = counts_calib.at[j, k].add(cd)
        
        P_test = (1.0 + counts_test) / (n + 1.0)
        P_calib_j = (1.0 + counts_calib[j]) / (n + 1.0)
        
        # Only recompute the specific row j that was modified
        if agg_func == 'min':
            M_test = jnp.min(P_test)
            M_calib = M_calib.at[j].set(jnp.min(P_calib_j))
        elif agg_func == 'avg':
            M_test = jnp.mean(P_test)
            M_calib = M_calib.at[j].set(jnp.mean(P_calib_j))
        elif agg_func == 'median':
            M_test = jnp.median(P_test)
            M_calib = M_calib.at[j].set(jnp.median(P_calib_j))
            
        rank_test = jnp.sum(M_calib <= M_test)
        valid_curr = (1.0 + rank_test) / (n + 1.0) > alpha
        
        return (counts_test, counts_calib, M_calib, y_curr, valid_curr), width_contrib

    _, width_contribs = jax.lax.scan(scan_step, init_state, jnp.arange(2 * n * K))
    total_width = jnp.sum(width_contribs)
    
    return jnp.empty(0), jnp.empty(0), total_width, coverage

@partial(jax.jit, static_argnames=['n_1', 'alpha'])
def tb_min_efficient(calib_scores, preds_test, Y_true, n_1, alpha):
    n, K = calib_scores.shape
    n_2 = n - n_1
    S_ref = calib_scores[:n_1]
    S_agg = calib_scores[n_1:]

    S_ref_sorted = jnp.sort(S_ref, axis=0)

    def compute_base_rank(val, col):
        return n_1 - jnp.searchsorted(col, val, side='left')

    ranks_agg = jax.vmap(jax.vmap(compute_base_rank, in_axes=(0, None)), in_axes=(1, 1))(S_agg, S_ref_sorted).T
    P_agg = (1.0 + ranks_agg) / (n_1 + 1.0)
    
    M_agg = jnp.min(P_agg, axis=1)

    idx = jnp.floor(alpha * (n_2 + 1.0)).astype(jnp.int32)
    idx = jnp.maximum(0, idx)
    u_tb = jnp.sort(M_agg)[idx]

    idx_R = n_1 + 1 - jnp.round(u_tb * (n_1 + 1.0)).astype(jnp.int32)
    idx_R = jnp.clip(idx_R, 0, n_1 - 1)
    
    R_l = S_ref_sorted[idx_R, jnp.arange(K)]

    L = jnp.max(preds_test - R_l)
    U = jnp.min(preds_test + R_l)

    valid = L <= U
    width = jnp.where(valid, U - L, 0.0)
    coverage = jnp.where(valid & (Y_true >= L) & (Y_true <= U), 1.0, 0.0)

    return width, coverage

@partial(jax.jit, static_argnames=['agg_func', 'alpha'])
def sb_worst_case_prediction_set(calib_scores, preds_test, Y_true, alpha, agg_func='min'):
    n, K = calib_scores.shape
    
    if agg_func == 'min': wc_factor = K
    elif agg_func == 'avg': wc_factor = 2.0
    elif agg_func == 'median': wc_factor = 2.0
    threshold = alpha / wc_factor
    
    S_calib_sorted = jnp.sort(calib_scores, axis=0)
    
    # Exact Coverage on Y_true
    test_scores_true = jnp.abs(Y_true - preds_test)
    
    def compute_test_rank(val, col):
        return n - jnp.searchsorted(col, val, side='left')
        
    rank_eval_true = jax.vmap(compute_test_rank)(test_scores_true, S_calib_sorted.T)
    P_test_true = (1.0 + rank_eval_true) / (n + 1.0)
    
    if agg_func == 'min': M_true = jnp.min(P_test_true)
    elif agg_func == 'avg': M_true = jnp.mean(P_test_true)
    elif agg_func == 'median': M_true = jnp.median(P_test_true)
    
    coverage = jnp.where(M_true > threshold, 1.0, 0.0)
    
    # Extract Breakpoints and Sweep
    Y_up = preds_test[None, :] + calib_scores
    Y_down = preds_test[None, :] - calib_scores
    
    y_events = jnp.concatenate([Y_down.flatten(), Y_up.flatten()])
    k_events = jnp.concatenate([jnp.tile(jnp.arange(K), n), jnp.tile(jnp.arange(K), n)])
    test_deltas = jnp.concatenate([jnp.ones(n * K, dtype=jnp.int32), -jnp.ones(n * K, dtype=jnp.int32)])
    
    sort_idx = jnp.argsort(y_events)
    y_events = y_events[sort_idx]
    k_events = k_events[sort_idx]
    test_deltas = test_deltas[sort_idx]
    
    counts_test_init = jnp.zeros(K, dtype=jnp.int32)
    P_test_init = (1.0 + counts_test_init) / (n + 1.0)
    
    if agg_func == 'min': M_init = jnp.min(P_test_init)
    elif agg_func == 'avg': M_init = jnp.mean(P_test_init)
    elif agg_func == 'median': M_init = jnp.median(P_test_init)
    
    valid_init = M_init > threshold
    init_state = (counts_test_init, y_events[0], valid_init)
    
    def scan_step(state, event_idx):
        counts_test, y_prev, valid_prev = state
        
        y_curr = y_events[event_idx]
        k = k_events[event_idx]
        td = test_deltas[event_idx]
        
        width_contrib = jnp.where(valid_prev, y_curr - y_prev, 0.0)
        
        counts_test = counts_test.at[k].add(td)
        P_test = (1.0 + counts_test) / (n + 1.0)
        
        if agg_func == 'min': M_curr = jnp.min(P_test)
        elif agg_func == 'avg': M_curr = jnp.mean(P_test)
        elif agg_func == 'median': M_curr = jnp.median(P_test)
        
        valid_curr = M_curr > threshold
        return (counts_test, y_curr, valid_curr), width_contrib

    _, width_contribs = jax.lax.scan(scan_step, init_state, jnp.arange(2 * n * K))
    total_width = jnp.sum(width_contribs)
    
    return jnp.empty(0), jnp.empty(0), total_width, coverage

@partial(jax.jit, static_argnames=['n_1', 'agg_func', 'alpha'])
def tb_worst_case_prediction_set(calib_scores, preds_test, Y_true, n_1, alpha, agg_func='min'):
    n, K = calib_scores.shape
    S_ref = calib_scores[:n_1]
    
    if agg_func == 'min': wc_factor = K
    elif agg_func == 'avg': wc_factor = 2.0
    elif agg_func == 'median': wc_factor = 2.0
    threshold = alpha / wc_factor
    
    S_ref_sorted = jnp.sort(S_ref, axis=0)
    
    # Exact Coverage on Y_true
    test_scores_true = jnp.abs(Y_true - preds_test)
    
    def compute_test_rank(val, col):
        return n_1 - jnp.searchsorted(col, val, side='left')
        
    rank_eval_true = jax.vmap(compute_test_rank)(test_scores_true, S_ref_sorted.T)
    P_test_true = (1.0 + rank_eval_true) / (n_1 + 1.0)
    
    if agg_func == 'min': M_true = jnp.min(P_test_true)
    elif agg_func == 'avg': M_true = jnp.mean(P_test_true)
    elif agg_func == 'median': M_true = jnp.median(P_test_true)
    
    coverage = jnp.where(M_true >= threshold, 1.0, 0.0)
    
    # Extract Breakpoints and Sweep
    Y_up = preds_test[None, :] + S_ref_sorted
    Y_down = preds_test[None, :] - S_ref_sorted
    
    y_events = jnp.concatenate([Y_down.flatten(), Y_up.flatten()])
    k_events = jnp.concatenate([jnp.tile(jnp.arange(K), n_1), jnp.tile(jnp.arange(K), n_1)])
    test_deltas = jnp.concatenate([jnp.ones(n_1 * K, dtype=jnp.int32), -jnp.ones(n_1 * K, dtype=jnp.int32)])
    
    sort_idx = jnp.argsort(y_events)
    y_events = y_events[sort_idx]
    k_events = k_events[sort_idx]
    test_deltas = test_deltas[sort_idx]
    
    counts_test_init = jnp.zeros(K, dtype=jnp.int32)
    P_test_init = (1.0 + counts_test_init) / (n_1 + 1.0)
    
    if agg_func == 'min': M_init = jnp.min(P_test_init)
    elif agg_func == 'avg': M_init = jnp.mean(P_test_init)
    elif agg_func == 'median': M_init = jnp.median(P_test_init)
    
    valid_init = M_init >= threshold
    init_state = (counts_test_init, y_events[0], valid_init)
    
    def scan_step(state, event_idx):
        counts_test, y_prev, valid_prev = state
        
        y_curr = y_events[event_idx]
        k = k_events[event_idx]
        td = test_deltas[event_idx]
        
        width_contrib = jnp.where(valid_prev, y_curr - y_prev, 0.0)
        
        counts_test = counts_test.at[k].add(td)
        P_test = (1.0 + counts_test) / (n_1 + 1.0)
        
        if agg_func == 'min': M_curr = jnp.min(P_test)
        elif agg_func == 'avg': M_curr = jnp.mean(P_test)
        elif agg_func == 'median': M_curr = jnp.median(P_test)
        
        valid_curr = M_curr >= threshold
        return (counts_test, y_curr, valid_curr), width_contrib

    _, width_contribs = jax.lax.scan(scan_step, init_state, jnp.arange(2 * n_1 * K))
    total_width = jnp.sum(width_contribs)
    
    return jnp.empty(0), jnp.empty(0), total_width, coverage

def generate_data_conformal(key, n_samples, K):
    k1, k2, k3, k4, k_shared = jax.random.split(key, 5)
    X = jax.random.uniform(k1, shape=(n_samples, 1), minval=-2.0, maxval=2.0)
    true_f = jnp.sin(3 * X)
    noise = jax.random.normal(k2, shape=(n_samples, 1)) * (1 + jnp.abs(X))
    Y = true_f + noise
    k_preds = jax.random.split(k4, K)
    preds = []

    shared_noise = jax.random.normal(k_shared, shape=(n_samples, 1))
    for i, kr in enumerate(k_preds):
        if i == 0:
            # Predictor 0 is highly accurate
            scale = 0.01
        else:
            # Predictors 1 to K-1 are highly inaccurate
            scale = 10
        independent_noise = jax.random.normal(kr, shape=(n_samples, 1))
        pred_noise = (0.99 * shared_noise + 0.01 * independent_noise) * scale
        preds.append(true_f + pred_noise)
        
    preds = jnp.concatenate(preds, axis=1)
    return X, Y, preds
