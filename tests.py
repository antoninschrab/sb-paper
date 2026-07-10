import inspect
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial


def wrap_stat(func):
    """
    Inspects a function and wraps it to consistently accept (key, data).
    If the function only requires 'data', the 'key' is discarded.
    """
    try:
        sig = inspect.signature(func)
        pos_params = [
            p for p in sig.parameters.values() 
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        required_params = [
            p for p in pos_params if p.default == inspect.Parameter.empty
        ]
        if len(required_params) == 1 or (len(required_params) == 0 and len(pos_params) > 0):
            return lambda k, d: func(d)
        if len(required_params) >= 2:
            return lambda k, d: func(k, d)
    except ValueError:
        pass
    return lambda k, d: func(d)

    
def sb_test(key, data, multi_stats, framework='two sample', B=499, alpha=0.05, break_ties=True):
    """
    Generalized SB test.
    """
    B_plus_1 = B + 1
    keys = jax.random.split(key, 5)
    key_block0, key_block1, key_block2, key_block3, _ = keys

    # Transforms and Statistics
    
    if callable(multi_stats):
        apply_funcs = wrap_stat(multi_stats)
    else:
        wrapped_funcs = [wrap_stat(f) for f in multi_stats]
        def apply_funcs(k, item):
            subkeys = jax.random.split(k, len(wrapped_funcs))
            return jnp.stack([func(sk, item) for func, sk in zip(wrapped_funcs, subkeys)])

    X, Y = data
    if framework in ('two sample', 'two-sample'):
        m = X.shape[0]
        Z = jnp.concatenate((X, Y), axis=0) # Pooled data
        def transform_fn(k, d):
            permuted_Z = jax.random.permutation(k, Z, axis=0)
            return permuted_Z[:m], permuted_Z[m:]
    elif framework in ('independence', 'ind'):
        def transform_fn(k, d):
            _X, _Y = d
            permuted_Y = jax.random.permutation(k, _Y, axis=0)
            return _X, permuted_Y
    else:
        raise ValueError("Invalid framework. Use 'two sample' or 'independence'.")

    key_orig, key_loop = jax.random.split(key_block0)
    original_stat = apply_funcs(key_orig, data)

    def scan_step(carry, current_key):
        key_t, key_s = jax.random.split(current_key)
        permuted_data = transform_fn(key_t, data)
        stat = apply_funcs(key_s, permuted_data)
        return carry, stat

    loop_keys = jax.random.split(key_loop, B)
    _, permuted_stats = jax.lax.scan(scan_step, init=None, xs=loop_keys)

    original_stat_expanded = jnp.expand_dims(original_stat, axis=0)
    result_0 = jnp.concatenate([original_stat_expanded, permuted_stats], axis=0)

    # P-value matrix computation from statistics matrix
    
    def col_rank_stats_fn(k, col):
        if break_ties:
            U = jax.random.uniform(k, shape=col.shape)
            sort_indices = jnp.lexsort((U, -col))
        else:
            sort_indices = jnp.argsort(-col)
            
        ranks = jnp.argsort(sort_indices)
        return (ranks + 1) / B_plus_1

    if result_0.ndim == 1:
        result_1 = col_rank_stats_fn(key_block1, result_0)
    else:
        result_1 = jax.vmap(col_rank_stats_fn, in_axes=(None, 1), out_axes=1)(key_block1, result_0)
    
    # Minimum Merging Function
    
    def row_min_fn(k, row):
        return jnp.min(row)
    
    row_keys = jax.random.split(key_block2, B_plus_1)
    result_2 = jax.vmap(row_min_fn)(row_keys, result_1)

    # Final p-value computation
    
    X_flat = jnp.ravel(result_2)
    B_plus_1_final = X_flat.shape[0]
    x_0 = X_flat[0]
    
    if break_ties:
        U_final = jax.random.uniform(key_block3, shape=X_flat.shape)
        u_0 = U_final[0]
        
        strict_less = X_flat < x_0
        tie_breaker = (X_flat == x_0) & (U_final <= u_0)
        rank_1_indexed = jnp.sum(strict_less | tie_breaker)
    else:
        # Conservative p-value formulation strictly retains ties
        rank_1_indexed = jnp.sum(X_flat <= x_0)
    
    p_value = rank_1_indexed / B_plus_1_final
    
    return p_value, (p_value <= alpha)


def tb_test(key, data, multi_stats, framework='two sample', B=499, alpha=0.05, break_ties=True):
    """
    Generalized TB test.
    Computes 2*B transformations automatically.
    """
    B_double = B * 2 
    keys = jax.random.split(key, 5)
    key_block0, key_block1, key_block2, key_block3, _ = keys

    # Transforms and Statistics (2B)
    
    if callable(multi_stats):
        apply_funcs = wrap_stat(multi_stats)
    else:
        wrapped_funcs = [wrap_stat(f) for f in multi_stats]
        def apply_funcs(k, item):
            subkeys = jax.random.split(k, len(wrapped_funcs))
            return jnp.stack([func(sk, item) for func, sk in zip(wrapped_funcs, subkeys)])

    X, Y = data
    if framework in ('two sample', 'two-sample'):
        m = X.shape[0]
        Z = jnp.concatenate((X, Y), axis=0) # Pooled data
        def transform_fn(k, d):
            permuted_Z = jax.random.permutation(k, Z, axis=0)
            return permuted_Z[:m], permuted_Z[m:]
    elif framework in ('independence', 'ind'):
        def transform_fn(k, d):
            _X, _Y = d
            permuted_Y = jax.random.permutation(k, _Y, axis=0)
            return _X, permuted_Y
    else:
        raise ValueError("Invalid framework. Use 'two sample' or 'independence'.")

    key_orig, key_loop = jax.random.split(key_block0)
    original_stat = apply_funcs(key_orig, data)

    def scan_step(carry, current_key):
        key_t, key_s = jax.random.split(current_key)
        permuted_data = transform_fn(key_t, data)
        stat = apply_funcs(key_s, permuted_data)
        return carry, stat

    loop_keys = jax.random.split(key_loop, B_double)
    _, permuted_stats = jax.lax.scan(scan_step, init=None, xs=loop_keys)

    original_stat_expanded = jnp.expand_dims(original_stat, axis=0)
    result_0 = jnp.concatenate([original_stat_expanded, permuted_stats], axis=0)

    # P-value matrix computation from statistics matrix TB
    
    def col_rank_stats_TB_fn(k, col):
        C = B 
        N = col.shape[0] - C  # N = B + 1
        
        T_b = col[:N]  # (B+1,)
        T_c = col[N:]  # (B,)
        
        # TB Test evaluates T_c > T_b
        indicator = T_c[None, :] > T_b[:, None]
        sum_indicator = jnp.sum(indicator, axis=1)
        
        p_vals = (1.0 + sum_indicator) / (C + 1.0)
        return p_vals

    if result_0.ndim == 1:
        result_1 = col_rank_stats_TB_fn(key_block1, result_0)
    else:
        result_1 = jax.vmap(col_rank_stats_TB_fn, in_axes=(None, 1), out_axes=1)(key_block1, result_0)
    
    # Minimum Merging Function
    
    def row_min_fn(k, row):
        return jnp.min(row)
    
    row_keys = jax.random.split(key_block2, B + 1) 
    result_2 = jax.vmap(row_min_fn)(row_keys, result_1)

    # Final p-value computation
    
    X_flat = jnp.ravel(result_2)
    B_plus_1_final = X_flat.shape[0]
    x_0 = X_flat[0]
    
    if break_ties:
        U_final = jax.random.uniform(key_block3, shape=X_flat.shape)
        u_0 = U_final[0]
        strict_less = X_flat < x_0
        tie_breaker = (X_flat == x_0) & (U_final <= u_0)
        rank_1_indexed = jnp.sum(strict_less | tie_breaker)
    else:
        # Conservative p-value formulation strictly retains ties
        rank_1_indexed = jnp.sum(X_flat <= x_0)
    
    p_value = rank_1_indexed / B_plus_1_final
    
    return p_value, (p_value <= alpha)


def classical_permutation_test(key, data, single_stat, framework='two sample', B=499, alpha=0.05, break_ties=True):
    """
    Classical Permutation Test.
    """
    B_plus_1 = B + 1
    
    keys = jax.random.split(key, 2)
    key_block0, key_block1 = keys

    # Transforms and Statistics

    if callable(single_stat):
        apply_funcs = wrap_stat(single_stat)
    else:
        wrapped_funcs = [wrap_stat(f) for f in single_stat]
        def apply_funcs(k, item):
            subkeys = jax.random.split(k, len(wrapped_funcs))
            return jnp.stack([func(sk, item) for func, sk in zip(wrapped_funcs, subkeys)])

    # Setup the transformation behavior based on the framework
    if framework in ('one sample test zero', 'one-sample'):
        X = data
        m = X.shape[0]
        def transform_fn(k, d):
            # One Rademacher variable per data point
            signs = jax.random.choice(k, jnp.array([-1.0, 1.0]), shape=(m, 1))
            return d * signs
            
    elif framework in ('two sample', 'two-sample'):
        X, Y = data
        m = X.shape[0]
        Z = jnp.concatenate((X, Y), axis=0) # Pooled data
        def transform_fn(k, d):
            # permute the pooled sample
            permuted_Z = jax.random.permutation(k, Z, axis=0)
            return permuted_Z[:m], permuted_Z[m:]
            
    elif framework in ('independence', 'ind'):
        X, Y = data
        def transform_fn(k, d):
            # permute Y
            _X, _Y = d
            permuted_Y = jax.random.permutation(k, _Y, axis=0) # Only permute Y
            return _X, permuted_Y
            
    else:
        raise ValueError("Invalid framework. Use 'two sample', 'independence', or 'one sample test zero'.")

    # Evaluate on the original data
    key_orig, key_loop = jax.random.split(key_block0)
    original_stat = apply_funcs(key_orig, data)

    # Fast sequential permutation loop
    def scan_step(carry, current_key):
        key_t, key_s = jax.random.split(current_key)
        permuted_data = transform_fn(key_t, data)
        stat = apply_funcs(key_s, permuted_data)
        return carry, stat

    loop_keys = jax.random.split(key_loop, B)
    _, permuted_stats = jax.lax.scan(scan_step, init=None, xs=loop_keys)

    original_stat_expanded = jnp.expand_dims(original_stat, axis=0)
    result_0 = jnp.concatenate([original_stat_expanded, permuted_stats], axis=0)

    # Final p-value computation from statistics

    X_flat = jnp.ravel(-result_0)
    B_plus_1_final = X_flat.shape[0]
    x_0 = X_flat[0]
    
    if break_ties:
        U_final = jax.random.uniform(key_block1, shape=X_flat.shape)
        u_0 = U_final[0]
        strict_less = X_flat < x_0
        tie_breaker = (X_flat == x_0) & (U_final <= u_0)
        rank_1_indexed = jnp.sum(strict_less | tie_breaker)
    else:
        # Conservative p-value formulation strictly retains ties
        rank_1_indexed = jnp.sum(X_flat <= x_0)
    
    p_value = rank_1_indexed / B_plus_1_final
    
    return p_value, (p_value <= alpha)


def hsic_sb_test(
    key, 
    data, 
    bandwidth_min_X, 
    bandwidth_max_X, 
    number_bandwidths_X,
    bandwidth_min_Y, 
    bandwidth_max_Y, 
    number_bandwidths_Y,
    B=499, 
    alpha=0.05,
    break_ties=True
):
    """
    HSIC SB Test (V-statistic, Gaussian Kernel, Permutations).
    """
    X, Y = data
    n = X.shape[0]
    B_plus_1 = B + 1
    
    keys = jax.random.split(key, 4)
    key_perms, key_block1, key_block3, _ = keys

    # Compute distances and bandwidth collection

    def compute_sq_l2(A):
        sq_norms = jnp.sum(A ** 2, axis=-1)
        D_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * jnp.dot(A, A.T)
        return jnp.clip(D_sq, min=0.0)

    D_sq_X = compute_sq_l2(X)
    D_sq_Y = compute_sq_l2(Y)
    
    # Generate geometric discretisation for bandwidths
    bws_X = jnp.geomspace(bandwidth_min_X, bandwidth_max_X, number_bandwidths_X)
    bws_Y = jnp.geomspace(bandwidth_min_Y, bandwidth_max_Y, number_bandwidths_Y)
    
    # Generate permutation indices (Only permute Y for independence test)
    keys_perm = jax.random.split(key_perms, B + 1)
    idx = jax.vmap(jax.random.permutation, in_axes=(0, None))(keys_perm, jnp.arange(n))
    idx = idx.at[0].set(jnp.arange(n))  # Index 0 is the unpermuted original data

    # Compute HSIC V-statistic

    def scan_X(carry_X, bw_X):
        # Gaussian Kernel for X: exp(-D_sq / (2 * bw^2))
        K_X = jnp.exp(-D_sq_X / (2.0 * bw_X ** 2))
        
        # Center kernel matrix K_X (H K_X H for H = I - 1 @ 1.T / n)
        K_X_centered = K_X - K_X.mean(axis=0) - K_X.mean(axis=1, keepdims=True) + K_X.mean()

        def scan_Y(carry_Y, bw_Y):
            # Gaussian Kernel for Y
            K_Y = jnp.exp(-D_sq_Y / (2.0 * bw_Y ** 2))

            def compute_hsic(index):
                # Permute K_Y row-wise and column-wise
                K_Y_perm = K_Y[index[:, None], index[None, :]]
                # HSIC V-statistic computation
                squared_hsic = jnp.sum(K_Y_perm * K_X_centered) / (n ** 2)
                return jnp.sqrt(jnp.clip(squared_hsic, min=0.0))

            hsic_values = jax.lax.map(compute_hsic, idx)
            return None, hsic_values

        _, hsic_row = jax.lax.scan(scan_Y, None, bws_Y)
        return None, hsic_row

    # Run the nested scan over the bandwidth grids
    _, H_k_X_k_Y = jax.lax.scan(scan_X, None, bws_X)

    # Flatten the bandwidth dimensions to get (B+1, D), where D = number_bandwidths_X * number_bandwidths_Y
    result_0 = H_k_X_k_Y.reshape((-1, B_plus_1)).T

    # P-value matrix computation from statistics matrix

    def col_rank_stats_fn(k, col):
        if break_ties:
            U = jax.random.uniform(k, shape=col.shape)
            # Sort by -col (larger statistic -> smaller p-value rank)
            sort_indices = jnp.lexsort((U, -col))
        else:
            sort_indices = jnp.argsort(-col)
            
        ranks = jnp.argsort(sort_indices)
        return (ranks + 1) / B_plus_1

    if result_0.ndim == 1:
        result_1 = col_rank_stats_fn(key_block1, result_0)
    else:
        result_1 = jax.vmap(col_rank_stats_fn, in_axes=(None, 1), out_axes=1)(key_block1, result_0)

    # Minimum Merging Function

    # Take the minimum p-value across all bandwidth combinations for each permutation row
    result_2 = jnp.min(result_1, axis=1)

    # Final p-value computation

    X_flat = jnp.ravel(result_2)
    B_plus_1_final = X_flat.shape[0]
    x_0 = X_flat[0]
    
    if break_ties:
        U_final = jax.random.uniform(key_block3, shape=X_flat.shape)
        u_0 = U_final[0]
        strict_less = X_flat < x_0
        tie_breaker = (X_flat == x_0) & (U_final <= u_0)
        rank_1_indexed = jnp.sum(strict_less | tie_breaker)
    else:
        # Conservative p-value formulation strictly retains ties
        rank_1_indexed = jnp.sum(X_flat <= x_0)
        
    p_value = rank_1_indexed / B_plus_1_final
    output = (p_value <= alpha)
    return p_value, output

# =====================================================================
# MAX-T TEST
# =====================================================================

def maxT_test(key, data, multi_stats, framework='two sample', B=499, alpha=0.05, break_ties=True):
    """
    MaxT test using the exact analytical quantile computation.
    """
    def wrap_stat(func):
        try:
            sig = inspect.signature(func)
            pos_params = [
                p for p in sig.parameters.values() 
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            required_params = [
                p for p in pos_params if p.default == inspect.Parameter.empty
            ]
            if len(required_params) == 1 or (len(required_params) == 0 and len(pos_params) > 0):
                return lambda k, d: func(d)
            if len(required_params) >= 2:
                return lambda k, d: func(k, d)
        except ValueError:
            pass
        return lambda k, d: func(d)

    # Transforms and Statistics (2B+1, K matrix)

    if callable(multi_stats):
        apply_funcs = wrap_stat(multi_stats)
    else:
        wrapped_funcs = [wrap_stat(f) for f in multi_stats]
        def apply_funcs(k, item):
            subkeys = jax.random.split(k, len(wrapped_funcs))
            return jnp.stack([func(sk, item) for func, sk in zip(wrapped_funcs, subkeys)])

    X, Y = data
    if framework in ('two sample', 'two-sample'):
        m = X.shape[0]
        Z = jnp.concatenate((X, Y), axis=0) # Pooled data
        def transform_fn(k, d):
            permuted_Z = jax.random.permutation(k, Z, axis=0)
            return permuted_Z[:m], permuted_Z[m:]
    elif framework in ('independence', 'ind'):
        def transform_fn(k, d):
            _X, _Y = d
            permuted_Y = jax.random.permutation(k, _Y, axis=0)
            return _X, permuted_Y
    else:
        raise ValueError("Invalid framework. Use 'two sample' or 'independence'.")

    key_orig, key_loop = jax.random.split(key)
    original_stat = apply_funcs(key_orig, data)

    def scan_step(carry, current_key):
        key_t, key_s = jax.random.split(current_key)
        permuted_data = transform_fn(key_t, data)
        stat = apply_funcs(key_s, permuted_data)
        return carry, stat

    B_double = B * 2
    loop_keys = jax.random.split(key_loop, B_double)
    _, permuted_stats = jax.lax.scan(scan_step, init=None, xs=loop_keys)

    original_stat_expanded = jnp.expand_dims(original_stat, axis=0)
    result_0 = jnp.concatenate([original_stat_expanded, permuted_stats], axis=0) # (2B+1, K)

    # Transpose and split the matrix M
    M = result_0.T  # (K, 2B+1)
    K = M.shape[0]
    M_original = M[:, 0]
    M1_sorted = jnp.sort(M[:, :B + 1], axis=1)  # (K, B+1)
    M2 = M[:, B + 1:]                           # (K, B)
    
    # Exact Analytical Thresholding 
    # Calculate P_i for all B+1 columns simultaneously via broadcasting
    # M2: (K, B, 1), M1_sorted: (K, 1, B+1) -> diff: (K, B, B+1)
    diff = M2[:, :, None] - M1_sorted[:, None, :]
    
    # Max over K gives shape (B, B+1)
    max_diff = jnp.max(diff, axis=0)
    
    # Empirical probability for each possible quantile index in M1_sorted
    P_vals = jnp.mean(max_diff > 0, axis=0)  # (B+1,)

    # Find the smallest index (i_safe) where P <= alpha
    valid_mask = P_vals <= alpha
    
    # Append True to handle the case where no index satisfies the condition (u=0)
    valid_mask_padded = jnp.append(valid_mask, True) # (B+2,)
    i_safe = jnp.argmax(valid_mask_padded)

    # Evaluate Quantile and Output Result

    # Pad M1_sorted with -inf to handle the index shift (i_safe - 1)
    # This automatically assigns -inf when i_safe == 0, perfectly matching S_{(0)}
    neg_inf = jnp.full((K, 1), -jnp.inf)
    M1_padded = jnp.concatenate([neg_inf, M1_sorted], axis=1) # Shape: (K, B+2)

    # Need to drop the index by 1 (S_{(r-1)})
    # In the padded array, M1_padded[:, i_safe] is exactly M1_sorted[:, i_safe - 1]
    quantiles = M1_padded[:, i_safe]

    reject_stat_vals = M_original > quantiles

    # no p-value (only output) so return -1 for the p-value
    return -1.0, jnp.any(reject_stat_vals)


def data_driven_sb_test(key, data, multi_stats, framework='two sample', B=499, alpha=0.05, break_ties=True):
    """
    Data-Driven SB Test.
    """
    B_plus_1 = B + 1
    keys = jax.random.split(key, 6)
    key_block0, key_block1, key_block2, key_block3, key_block4, key_block5 = keys

    # Transforms and Statistics

    if callable(multi_stats):
        apply_funcs = wrap_stat(multi_stats)
    else:
        wrapped_funcs = [wrap_stat(f) for f in multi_stats]
        def apply_funcs(k, item):
            subkeys = jax.random.split(k, len(wrapped_funcs))
            return jnp.stack([func(sk, item) for func, sk in zip(wrapped_funcs, subkeys)])

    # Setup the transformation behavior based on the framework
    X, Y = data
    if framework in ('two sample', 'two-sample'):
        m = X.shape[0]
        Z = jnp.concatenate((X, Y), axis=0) # Pooled data
        def transform_fn(k, d):
            permuted_Z = jax.random.permutation(k, Z, axis=0)
            return permuted_Z[:m], permuted_Z[m:]
    elif framework in ('independence', 'ind'):
        def transform_fn(k, d):
            _X, _Y = d
            permuted_Y = jax.random.permutation(k, _Y, axis=0) # Only permute Y
            return _X, permuted_Y
    else:
        raise ValueError("Invalid framework. Use 'two sample' or 'independence'.")

    # Evaluate on the original data
    key_orig, key_loop = jax.random.split(key_block0)
    original_stat = apply_funcs(key_orig, data)

    # Fast sequential permutation loop
    def scan_step(carry, current_key):
        key_t, key_s = jax.random.split(current_key)
        permuted_data = transform_fn(key_t, data)
        stat = apply_funcs(key_s, permuted_data)
        return carry, stat

    loop_keys = jax.random.split(key_loop, B)
    _, permuted_stats = jax.lax.scan(scan_step, init=None, xs=loop_keys)

    original_stat_expanded = jnp.expand_dims(original_stat, axis=0)
    result_0 = jnp.concatenate([original_stat_expanded, permuted_stats], axis=0)

    # P-value matrix computation from statistics matrix

    def col_rank_stats_fn(k, col):
        if break_ties:
            U = jax.random.uniform(k, shape=col.shape)
            # Sort by -col (larger statistic -> smaller p-value)
            sort_indices = jnp.lexsort((U, -col))
        else:
            sort_indices = jnp.argsort(-col)
        ranks = jnp.argsort(sort_indices)
        return (ranks + 1) / B_plus_1

    if result_0.ndim == 1:
        result_1 = col_rank_stats_fn(key_block1, result_0)
    else:
        result_1 = jax.vmap(col_rank_stats_fn, in_axes=(None, 1), out_axes=1)(key_block1, result_0)

    # Merging functions: Min, Avg, Median

    def row_mean_min_fn(row):
        return jnp.stack([row.size * jnp.min(row), 2 * jnp.mean(row), 2 * jnp.median(row)])
    
    result_2 = jax.vmap(row_mean_min_fn)(result_1)  # (B+1, 3)

    # P-value matrix computation from p-value matrix

    def col_rank_pvals_fn(k, col):
        if break_ties:
            U = jax.random.uniform(k, shape=col.shape)
            # Sort by col (smaller p-value -> smaller aggregated p-value)
            sort_indices = jnp.lexsort((U, col))
        else:
            sort_indices = jnp.argsort(col)
            
        ranks = jnp.argsort(sort_indices)
        return (ranks + 1) / B_plus_1

    if result_2.ndim == 1:
        result_3 = col_rank_pvals_fn(key_block3, result_2)
    else:
        result_3 = jax.vmap(col_rank_pvals_fn, in_axes=(None, 1), out_axes=1)(key_block3, result_2)

    # Minimum Merging Function

    def row_min_fn(row):
        return jnp.min(row)
    
    if result_3.ndim > 1:
        result_4 = jax.vmap(row_min_fn)(result_3)
    else:
        result_4 = result_3

    # Compute final p-value
    X_flat = jnp.ravel(result_4)
    B_plus_1_final = X_flat.shape[0]
    x_0 = X_flat[0]
    
    if break_ties:
        U_final = jax.random.uniform(key_block5, shape=X_flat.shape)
        u_0 = U_final[0]
        strict_less = X_flat < x_0
        tie_breaker = (X_flat == x_0) & (U_final <= u_0)
        rank_1_indexed = jnp.sum(strict_less | tie_breaker)
    else:
        # Conservative p-value formulation strictly retains ties
        rank_1_indexed = jnp.sum(X_flat <= x_0)
    
    p_value = rank_1_indexed / B_plus_1_final
    
    return p_value, (p_value <= alpha)


def sb_worst_case_tests(key, data, multi_stats, framework='two sample', B=499, alpha=0.05, break_ties=True):
    """
    Data-Driven SB Test with Worst-Case and SB.
    """
    B_plus_1 = B + 1
    keys = jax.random.split(key, 6)
    key_block0, key_block1, key_block2, key_block3, key_block4, key_block5 = keys

    # Transforms and Statistics

    if callable(multi_stats):
        apply_funcs = wrap_stat(multi_stats)
    else:
        wrapped_funcs = [wrap_stat(f) for f in multi_stats]
        def apply_funcs(k, item):
            subkeys = jax.random.split(k, len(wrapped_funcs))
            return jnp.stack([func(sk, item) for func, sk in zip(wrapped_funcs, subkeys)])

    # Setup the transformation behavior based on the framework
    if framework in ('one sample test zero', 'one-sample'):
        X = data
        m = X.shape[0]
        def transform_fn(k, d):
            signs = jax.random.choice(k, jnp.array([-1.0, 1.0]), shape=(m, 1))
            return d * signs
            
    elif framework in ('two sample', 'two-sample'):
        X, Y = data
        m = X.shape[0]
        Z = jnp.concatenate((X, Y), axis=0) # Pooled data
        def transform_fn(k, d):
            permuted_Z = jax.random.permutation(k, Z, axis=0)
            return permuted_Z[:m], permuted_Z[m:]
            
    elif framework in ('independence', 'ind'):
        X, Y = data
        def transform_fn(k, d):
            _X, _Y = d
            permuted_Y = jax.random.permutation(k, _Y, axis=0) # Only permute Y
            return _X, permuted_Y
            
    else:
        raise ValueError("Invalid framework. Use 'two sample', 'independence', or 'one sample test zero'.")

    # Evaluate on the original data
    key_orig, key_loop = jax.random.split(key_block0)
    original_stat = apply_funcs(key_orig, data)

    # Fast sequential permutation loop
    def scan_step(carry, current_key):
        key_t, key_s = jax.random.split(current_key)
        permuted_data = transform_fn(key_t, data)
        stat = apply_funcs(key_s, permuted_data)
        return carry, stat

    loop_keys = jax.random.split(key_loop, B)
    _, permuted_stats = jax.lax.scan(scan_step, init=None, xs=loop_keys)

    original_stat_expanded = jnp.expand_dims(original_stat, axis=0)
    result_0 = jnp.concatenate([original_stat_expanded, permuted_stats], axis=0)

    # P-value matrix computation from statistics matrix

    def col_rank_stats_fn(k, col):
        if break_ties:
            U = jax.random.uniform(k, shape=col.shape)
            # Sort by -col (larger statistic -> smaller p-value rank)
            sort_indices = jnp.lexsort((U, -col))
        else:
            sort_indices = jnp.argsort(-col)
            
        ranks = jnp.argsort(sort_indices)
        return (ranks + 1) / B_plus_1

    if result_0.ndim == 1:
        result_1 = col_rank_stats_fn(key_block1, result_0)
    else:
        result_1 = jax.vmap(col_rank_stats_fn, in_axes=(None, 1), out_axes=1)(key_block1, result_0)

    # Merging functions: Min, Avg, Median

    def row_mean_min_fn(row):
        return jnp.stack([row.size * jnp.min(row), 2 * jnp.mean(row), 2 * jnp.median(row)])
    
    # Vectorize row-wise feature engineering (Shape: (B+1, 3))
    result_2 = jax.vmap(row_mean_min_fn)(result_1)  
    B_plus_1_final = result_2.shape[0]

    # Extract strictly theoretical Worst-Case p-values (uncallibrated, capped at 1.0)
    p_worst_case_min, p_worst_case_mean, p_worst_case_med = jnp.minimum(1.0, result_2[0])

    # Calculate Single-Step SB p-values (calibrated via permutation ranks)
    def compute_sb_rank(col):
        x_0 = col[0]
        if break_ties:
            # same randomness across columns
            strict_less = col < x_0
            tie_breaker = (col == x_0) & (U_final <= u_0)
            return jnp.sum(strict_less | tie_breaker) / B_plus_1_final
        else:
            return jnp.sum(col <= x_0) / B_plus_1_final

    if break_ties:
        U_final = jax.random.uniform(key_block2, shape=(B_plus_1_final,))
        u_0 = U_final[0]

    # Map the rank computation over the 3 columns (min, mean, med)
    p_sb_min, p_sb_mean, p_sb_med = jax.vmap(compute_sb_rank, in_axes=1)(result_2)

    # Compute p-value matrix from p-value matrix

    def col_rank_pvals_fn(k, col):
        if break_ties:
            U = jax.random.uniform(k, shape=col.shape)
            # Sort by col (smaller p-value -> smaller aggregated p-value)
            sort_indices = jnp.lexsort((U, col))
        else:
            sort_indices = jnp.argsort(col)
            
        ranks = jnp.argsort(sort_indices)
        return (ranks + 1) / B_plus_1

    if result_2.ndim == 1:
        result_3 = col_rank_pvals_fn(key_block3, result_2)
    else:
        result_3 = jax.vmap(col_rank_pvals_fn, in_axes=(None, 1), out_axes=1)(key_block3, result_2)

    # Minimum Merging Function
    
    def row_min_fn(row):
        return jnp.min(row)
    
    if result_3.ndim > 1:
        result_4 = jax.vmap(row_min_fn)(result_3)
    else:
        result_4 = result_3

    # Compute final p-value
    
    X_flat = jnp.ravel(result_4)
    x_0 = X_flat[0]
    
    if break_ties:
        U_final_dd = jax.random.uniform(key_block5, shape=X_flat.shape)
        u_0_dd = U_final_dd[0]
        
        strict_less = X_flat < x_0
        tie_breaker = (X_flat == x_0) & (U_final_dd <= u_0_dd)
        rank_1_indexed = jnp.sum(strict_less | tie_breaker)
    else:
        rank_1_indexed = jnp.sum(X_flat <= x_0)
    
    p_data_driven_sb = rank_1_indexed / B_plus_1_final
    
    return (
        p_worst_case_min,  
        p_sb_min,  
        p_worst_case_med,  
        p_sb_med,
        p_worst_case_mean,  
        p_sb_mean,  
        p_data_driven_sb,  
    )


def tb_worst_case_tests(key, data, multi_stats, framework='two sample', B=499, alpha=0.05, break_ties=True):
    """
    Data-Driven TB Test with Worst-Case and TB.
    """
    B_plus_1 = B + 1
    B_double = B * 2
    keys = jax.random.split(key, 6)
    key_block0, key_block1, key_block2, key_block3, key_block4, key_block5 = keys

    # Transforms and Statistics (2B)

    if callable(multi_stats):
        apply_funcs = wrap_stat(multi_stats)
    else:
        wrapped_funcs = [wrap_stat(f) for f in multi_stats]
        def apply_funcs(k, item):
            subkeys = jax.random.split(k, len(wrapped_funcs))
            return jnp.stack([func(sk, item) for func, sk in zip(wrapped_funcs, subkeys)])

    # Setup the transformation behavior based on the framework
    if framework in ('one sample test zero', 'one-sample'):
        X = data
        m = X.shape[0]
        def transform_fn(k, d):
            signs = jax.random.choice(k, jnp.array([-1.0, 1.0]), shape=(m, 1))
            return d * signs
            
    elif framework in ('two sample', 'two-sample'):
        X, Y = data
        m = X.shape[0]
        Z = jnp.concatenate((X, Y), axis=0) # Pooled data
        def transform_fn(k, d):
            permuted_Z = jax.random.permutation(k, Z, axis=0)
            return permuted_Z[:m], permuted_Z[m:]
            
    elif framework in ('independence', 'ind'):
        X, Y = data
        def transform_fn(k, d):
            _X, _Y = d
            permuted_Y = jax.random.permutation(k, _Y, axis=0) # Only permute Y
            return _X, permuted_Y
            
    else:
        raise ValueError("Invalid framework. Use 'two sample', 'independence', or 'one sample test zero'.")

    # Evaluate on the original data
    key_orig, key_loop = jax.random.split(key_block0)
    original_stat = apply_funcs(key_orig, data)

    # Fast sequential permutation loop
    def scan_step(carry, current_key):
        key_t, key_s = jax.random.split(current_key)
        permuted_data = transform_fn(key_t, data)
        stat = apply_funcs(key_s, permuted_data)
        return carry, stat

    # TB test requires 2B permutations
    loop_keys = jax.random.split(key_loop, B_double)
    _, permuted_stats = jax.lax.scan(scan_step, init=None, xs=loop_keys)

    original_stat_expanded = jnp.expand_dims(original_stat, axis=0)
    result_0 = jnp.concatenate([original_stat_expanded, permuted_stats], axis=0)

    # P-value matrix computation from statistics matrix TB

    def col_rank_stats_TB_fn(k, col):
        N = B_plus_1
        C = col.shape[0] - N
        
        T_b = col[:N]  # (B+1,)
        T_c = col[N:]  # (B,)
        
        # TB Test evaluates T_c > T_b 
        indicator = T_c[None, :] > T_b[:, None]
        sum_indicator = jnp.sum(indicator, axis=1)
        
        p_vals = (1.0 + sum_indicator) / (C + 1.0)
        return p_vals

    if result_0.ndim == 1:
        result_1 = col_rank_stats_TB_fn(key_block1, result_0)
    else:
        result_1 = jax.vmap(col_rank_stats_TB_fn, in_axes=(None, 1), out_axes=1)(key_block1, result_0)

    # Merging functions: Min, Avg, Median

    def row_mean_min_fn(row):
        return jnp.stack([row.size * jnp.min(row), 2 * jnp.mean(row), 2 * jnp.median(row)])
    
    # Vectorize row-wise feature engineering (Shape: (B+1, 3))
    result_2 = jax.vmap(row_mean_min_fn)(result_1)  
    B_plus_1_final = result_2.shape[0]

    # Extract strictly theoretical Worst-Case p-values (uncallibrated, capped at 1.0)
    p_worst_case_min, p_worst_case_mean, p_worst_case_med = jnp.minimum(1.0, result_2[0])

    # Calculate p-values (calibrated via permutation ranks)
    def compute_rank(col):
        x_0 = col[0]
        if break_ties:
            # same randomness across columns
            strict_less = col < x_0
            tie_breaker = (col == x_0) & (U_final <= u_0)
            return jnp.sum(strict_less | tie_breaker) / B_plus_1_final
        else:
            return jnp.sum(col <= x_0) / B_plus_1_final

    if break_ties:
        U_final = jax.random.uniform(key_block2, shape=(B_plus_1_final,))
        u_0 = U_final[0]

    # Map the rank computation over the 3 columns (min, mean, med)
    p_sb_min, p_sb_mean, p_sb_med = jax.vmap(compute_rank, in_axes=1)(result_2)

    # Compute p-value matrix from p-value matrix

    def col_rank_pvals_fn(k, col):
        if break_ties:
            U = jax.random.uniform(k, shape=col.shape)
            # Sort by col (smaller p-value -> smaller aggregated p-value)
            sort_indices = jnp.lexsort((U, col))
        else:
            sort_indices = jnp.argsort(col)
            
        ranks = jnp.argsort(sort_indices)
        return (ranks + 1) / B_plus_1

    if result_2.ndim == 1:
        result_3 = col_rank_pvals_fn(key_block3, result_2)
    else:
        result_3 = jax.vmap(col_rank_pvals_fn, in_axes=(None, 1), out_axes=1)(key_block3, result_2)

    # Minimum Merging Function
    
    def row_min_fn(row):
        return jnp.min(row)
    
    if result_3.ndim > 1:
        result_4 = jax.vmap(row_min_fn)(result_3)
    else:
        result_4 = result_3

    # Compute Final P-value
    
    X_flat = jnp.ravel(result_4)
    x_0 = X_flat[0]
    
    if break_ties:
        U_final_dd = jax.random.uniform(key_block5, shape=X_flat.shape)
        u_0_dd = U_final_dd[0]
        strict_less = X_flat < x_0
        tie_breaker = (X_flat == x_0) & (U_final_dd <= u_0_dd)
        rank_1_indexed = jnp.sum(strict_less | tie_breaker)
    else:
        rank_1_indexed = jnp.sum(X_flat <= x_0)
    
    p_data_driven_sb = rank_1_indexed / B_plus_1_final
    
    return (
        p_worst_case_min,  
        p_sb_min,  
        p_worst_case_med,  
        p_sb_med,
        p_worst_case_mean,  
        p_sb_mean,  
        p_data_driven_sb,  
    )


def hsic_sb_worst_case_tests(
    key, 
    data, 
    bandwidth_min_X, 
    bandwidth_max_X, 
    number_bandwidths_X,
    bandwidth_min_Y, 
    bandwidth_max_Y, 
    number_bandwidths_Y,
    B=499, 
    alpha=0.05,
    break_ties=True,
    decreasing_collection=False,
    s=1.5,
    start_val=0.1,
):
    """
    HSIC Data-Driven SB Test with Worst-Case and SB.
    Supports standard geometric space or a decreasing collection grid.
    """
    X, Y = data
    n = X.shape[0]
    B_plus_1 = B + 1
    keys = jax.random.split(key, 6)
    key_perms, key_block1, key_block2, key_block3, key_block4, key_block5 = keys

    # Compute distances and collection of bandwidths
    
    def compute_sq_l2(A):
        sq_norms = jnp.sum(A ** 2, axis=-1)
        D_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * jnp.dot(A, A.T)
        return jnp.clip(D_sq, min=0.0)

    D_sq_X = compute_sq_l2(X)
    D_sq_Y = compute_sq_l2(Y)
    
    # Grid generation logic based on the decreasing_collection flag
    if decreasing_collection:
        k_X = jnp.arange(1, number_bandwidths_X + 1)
        bws_X = start_val * s ** (1 - k_X)
        
        k_Y = jnp.arange(1, number_bandwidths_Y + 1)
        bws_Y = start_val * s ** (1 - k_Y)
    else:
        bws_X = jnp.geomspace(bandwidth_min_X, bandwidth_max_X, number_bandwidths_X)
        bws_Y = jnp.geomspace(bandwidth_min_Y, bandwidth_max_Y, number_bandwidths_Y)
    
    keys_perm = jax.random.split(key_perms, B + 1)
    idx = jax.vmap(jax.random.permutation, in_axes=(0, None))(keys_perm, jnp.arange(n))
    idx = idx.at[0].set(jnp.arange(n))  # Index 0 is the unpermuted original data

    # Compute HSIC V-statistics
    
    def scan_X(carry_X, bw_X):
        K_X = jnp.exp(-D_sq_X / (2.0 * bw_X ** 2))
        K_X_centered = K_X - K_X.mean(axis=0) - K_X.mean(axis=1, keepdims=True) + K_X.mean()

        def scan_Y(carry_Y, bw_Y):
            K_Y = jnp.exp(-D_sq_Y / (2.0 * bw_Y ** 2))

            def compute_hsic(index):
                K_Y_perm = K_Y[index[:, None], index[None, :]]
                squared_hsic = jnp.sum(K_Y_perm * K_X_centered) / (n ** 2)
                return jnp.sqrt(jnp.clip(squared_hsic, min=0.0))

            hsic_values = jax.lax.map(compute_hsic, idx)
            return None, hsic_values

        _, hsic_row = jax.lax.scan(scan_Y, None, bws_Y)
        return None, hsic_row

    _, H_k_X_k_Y = jax.lax.scan(scan_X, None, bws_X)
    result_0 = H_k_X_k_Y.reshape((-1, B_plus_1)).T

    # P-value matrix computation from statistics matrix
    
    def col_rank_stats_fn(k, col):
        if break_ties:
            U = jax.random.uniform(k, shape=col.shape)
            sort_indices = jnp.lexsort((U, -col))
        else:
            sort_indices = jnp.argsort(-col)
            
        ranks = jnp.argsort(sort_indices)
        return (ranks + 1) / B_plus_1

    if result_0.ndim == 1:
        result_1 = col_rank_stats_fn(key_block1, result_0)
    else:
        result_1 = jax.vmap(col_rank_stats_fn, in_axes=(None, 1), out_axes=1)(key_block1, result_0)

    # Merging Functions: Min, Average, Median
    
    def row_mean_min_fn(row):
        return jnp.stack([row.size * jnp.min(row), 2 * jnp.mean(row), 2 * jnp.median(row)])
    
    result_2 = jax.vmap(row_mean_min_fn)(result_1)  
    B_plus_1_final = result_2.shape[0]

    p_worst_case_min, p_worst_case_mean, p_worst_case_med = jnp.minimum(1.0, result_2[0])

    # Calculate p-values (calibrated via permutation ranks)
    def compute_sb_rank(col):
        x_0 = col[0]
        if break_ties:
            strict_less = col < x_0
            tie_breaker = (col == x_0) & (U_final <= u_0)
            return jnp.sum(strict_less | tie_breaker) / B_plus_1_final
        else:
            return jnp.sum(col <= x_0) / B_plus_1_final

    if break_ties:
        U_final = jax.random.uniform(key_block2, shape=(B_plus_1_final,))
        u_0 = U_final[0]

    p_sb_min, p_sb_mean, p_sb_med = jax.vmap(compute_sb_rank, in_axes=1)(result_2)

    #  Compute p-value matrix from p-value matrix

    def col_rank_pvals_fn(k, col):
        if break_ties:
            U = jax.random.uniform(k, shape=col.shape)
            sort_indices = jnp.lexsort((U, col))
        else:
            sort_indices = jnp.argsort(col)
            
        ranks = jnp.argsort(sort_indices)
        return (ranks + 1) / B_plus_1

    if result_2.ndim == 1:
        result_3 = col_rank_pvals_fn(key_block3, result_2)
    else:
        result_3 = jax.vmap(col_rank_pvals_fn, in_axes=(None, 1), out_axes=1)(key_block3, result_2)

    # Minimum Merging Function
    
    def row_min_fn(row):
        return jnp.min(row)
    
    if result_3.ndim > 1:
        result_4 = jax.vmap(row_min_fn)(result_3)
    else:
        result_4 = result_3

    # Compute Final P-value
    
    X_flat = jnp.ravel(result_4)
    x_0 = X_flat[0]
    
    if break_ties:
        U_final_dd = jax.random.uniform(key_block5, shape=X_flat.shape)
        u_0_dd = U_final_dd[0]
        
        strict_less = X_flat < x_0
        tie_breaker = (X_flat == x_0) & (U_final_dd <= u_0_dd)
        rank_1_indexed = jnp.sum(strict_less | tie_breaker)
    else:
        rank_1_indexed = jnp.sum(X_flat <= x_0)
    
    p_data_driven_sb = rank_1_indexed / B_plus_1_final
    
    return (
        p_worst_case_min,  
        p_sb_min,  
        p_worst_case_med,  
        p_sb_med,
        p_worst_case_mean,  
        p_sb_mean,  
        p_data_driven_sb,  
    )


def hsic_tb_worst_case_tests(
    key, 
    data, 
    bandwidth_min_X, 
    bandwidth_max_X, 
    number_bandwidths_X,
    bandwidth_min_Y, 
    bandwidth_max_Y, 
    number_bandwidths_Y,
    B=499, 
    alpha=0.05,
    break_ties=True,
    decreasing_collection=False,
    s=1.5,
    start_val=0.1,
):
    """
    HSIC Data-Driven TB Test with Worst-Case and TB.
    Supports standard geometric space or a decreasing collection grid.
    """
    X, Y = data
    n = X.shape[0]
    
    B_plus_1 = B + 1
    B_double = B * 2
    
    keys = jax.random.split(key, 6)
    key_perms, key_block1, key_block2, key_block3, key_block4, key_block5 = keys

    # Compute distances and collection of bandwidths
    
    def compute_sq_l2(A):
        sq_norms = jnp.sum(A ** 2, axis=-1)
        D_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * jnp.dot(A, A.T)
        return jnp.clip(D_sq, min=0.0)

    D_sq_X = compute_sq_l2(X)
    D_sq_Y = compute_sq_l2(Y)
    
    # Grid generation logic based on the decreasing_collection flag
    if decreasing_collection:
        k_X = jnp.arange(1, number_bandwidths_X + 1)
        bws_X = start_val * s ** (1 - k_X)
        
        k_Y = jnp.arange(1, number_bandwidths_Y + 1)
        bws_Y = start_val * s ** (1 - k_Y)
    else:
        bws_X = jnp.geomspace(bandwidth_min_X, bandwidth_max_X, number_bandwidths_X)
        bws_Y = jnp.geomspace(bandwidth_min_Y, bandwidth_max_Y, number_bandwidths_Y)
    
    # Generate 2B permutations for TB framework
    keys_perm = jax.random.split(key_perms, B_double + 1)
    idx = jax.vmap(jax.random.permutation, in_axes=(0, None))(keys_perm, jnp.arange(n))
    idx = idx.at[0].set(jnp.arange(n))

    # Compute HSIC V-statistics (2B)
    
    def scan_X(carry_X, bw_X):
        K_X = jnp.exp(-D_sq_X / (2.0 * bw_X ** 2))
        K_X_centered = K_X - K_X.mean(axis=0) - K_X.mean(axis=1, keepdims=True) + K_X.mean()

        def scan_Y(carry_Y, bw_Y):
            K_Y = jnp.exp(-D_sq_Y / (2.0 * bw_Y ** 2))

            def compute_hsic(index):
                K_Y_perm = K_Y[index[:, None], index[None, :]]
                squared_hsic = jnp.sum(K_Y_perm * K_X_centered) / (n ** 2)
                return jnp.sqrt(jnp.clip(squared_hsic, min=0.0))

            hsic_values = jax.lax.map(compute_hsic, idx)
            return None, hsic_values

        _, hsic_row = jax.lax.scan(scan_Y, None, bws_Y)
        return None, hsic_row

    _, H_k_X_k_Y = jax.lax.scan(scan_X, None, bws_X)
    
    # Reshape safely handles the 2B + 1 length
    result_0 = H_k_X_k_Y.reshape((-1, B_double + 1)).T

    # P-value matrix computation from statistics matrix TB
    
    def col_rank_stats_TB_fn(k, col):
        C = B 
        N = col.shape[0] - C  # N = B + 1
        
        T_b = col[:N]  # (B+1,)
        T_c = col[N:]  # (B,)
        
        indicator = T_c[None, :] > T_b[:, None]
        sum_indicator = jnp.sum(indicator, axis=1)
        
        p_vals = (1.0 + sum_indicator) / (C + 1.0)
        return p_vals

    if result_0.ndim == 1:
        result_1 = col_rank_stats_TB_fn(key_block1, result_0)
    else:
        result_1 = jax.vmap(col_rank_stats_TB_fn, in_axes=(None, 1), out_axes=1)(key_block1, result_0)

    # Merging Functions: Minimum, Average, Median
    
    def row_mean_min_fn(row):
        return jnp.stack([row.size * jnp.min(row), 2 * jnp.mean(row), 2 * jnp.median(row)])
    
    result_2 = jax.vmap(row_mean_min_fn)(result_1)  
    B_plus_1_final = result_2.shape[0]

    p_worst_case_min, p_worst_case_mean, p_worst_case_med = jnp.minimum(1.0, result_2[0])

    # Calculate p-values (calibrated via permutation ranks)
    def compute_rank(col):
        x_0 = col[0]
        if break_ties:
            strict_less = col < x_0
            tie_breaker = (col == x_0) & (U_final <= u_0)
            return jnp.sum(strict_less | tie_breaker) / B_plus_1_final
        else:
            return jnp.sum(col <= x_0) / B_plus_1_final

    if break_ties:
        U_final = jax.random.uniform(key_block2, shape=(B_plus_1_final,))
        u_0 = U_final[0]

    p_sb_min, p_sb_mean, p_sb_med = jax.vmap(compute_rank, in_axes=1)(result_2)

    # Compute p-value matrix from p-value matrix
    
    def col_rank_pvals_fn(k, col):
        if break_ties:
            U = jax.random.uniform(k, shape=col.shape)
            sort_indices = jnp.lexsort((U, col))
        else:
            sort_indices = jnp.argsort(col)
            
        ranks = jnp.argsort(sort_indices)
        return (ranks + 1) / B_plus_1

    if result_2.ndim == 1:
        result_3 = col_rank_pvals_fn(key_block3, result_2)
    else:
        result_3 = jax.vmap(col_rank_pvals_fn, in_axes=(None, 1), out_axes=1)(key_block3, result_2)

    # Minimum Merging Function
    
    def row_min_fn(row):
        return jnp.min(row)
    
    if result_3.ndim > 1:
        result_4 = jax.vmap(row_min_fn)(result_3)
    else:
        result_4 = result_3

    # Compute Final P-value
    
    X_flat = jnp.ravel(result_4)
    x_0 = X_flat[0]
    
    if break_ties:
        U_final_dd = jax.random.uniform(key_block5, shape=X_flat.shape)
        u_0_dd = U_final_dd[0]
        strict_less = X_flat < x_0
        tie_breaker = (X_flat == x_0) & (U_final_dd <= u_0_dd)
        rank_1_indexed = jnp.sum(strict_less | tie_breaker)
    else:
        rank_1_indexed = jnp.sum(X_flat <= x_0)
    
    p_data_driven_sb = rank_1_indexed / B_plus_1_final
    
    return (
        p_worst_case_min,  
        p_sb_min,  
        p_worst_case_med,  
        p_sb_med,
        p_worst_case_mean,  
        p_sb_mean,  
        p_data_driven_sb,  
    )


def seq_sb_test(stats_matrix, alpha=0.05, key=None):
    """
    Sequential SB Test.
    """
    B_plus_1, K = stats_matrix.shape
    alpha_j = alpha / K
    
    N_allow = jnp.floor(alpha_j * B_plus_1)
    
    # 1D array of tie-breakers (one per permutation)
    U = jax.random.uniform(key, shape=(B_plus_1,))
    
    # use the same randomness across columns
    U_matrix = jnp.broadcast_to(U[:, None], stats_matrix.shape)
    
    sort_indices = jnp.lexsort((U_matrix, -stats_matrix), axis=0)
    ranks = jnp.argsort(sort_indices, axis=0)
    p_vals = (ranks + 1.0) / B_plus_1
    
    m = jnp.minimum.accumulate(p_vals, axis=1)
    
    S_mask = jnp.ones(B_plus_1, dtype=bool)
    
    def stage_step(carry, j):
        S_mask, rejected, decision_stage = carry
        
        m_j = m[:, j]
        m_j_active = jnp.where(S_mask, m_j, 2.0)
        
        # U and m_j_active are both 1D, so lexsort works natively
        lex_sort_indices = jnp.lexsort((U, m_j_active))
        lex_ranks = jnp.argsort(lex_sort_indices)
        
        A_j_mask_raw = lex_ranks < N_allow
        A_j_mask = A_j_mask_raw & S_mask
        
        S_mask_next = S_mask & (~A_j_mask)
        
        just_rejected = A_j_mask[0] & (~rejected)
        rejected_next = rejected | A_j_mask[0]
        decision_stage_next = jnp.where(just_rejected, j + 1, decision_stage)
        
        return (S_mask_next, rejected_next, decision_stage_next), None

    init_carry = (S_mask, jnp.array(False), jnp.array(K))
    final_carry, _ = jax.lax.scan(stage_step, init_carry, jnp.arange(K))
    
    S_mask_final, rejected_final, decision_stage_final = final_carry
    return rejected_final, decision_stage_final
    

def generate_seq_data_matrix(key, n, d, epsilon, sigma, K, B, bandwidth):
    """
    Generate sequential data and compute the MMD U-statistic wild bootstrap 
    matrix of size (B+1, K) using a fixed Laplace kernel.
    """
    k_X, k_Y, k_wb, k_seq = jax.random.split(key, 4)
    
    X = jax.random.normal(k_X, shape=(n, d))
    Y_1 = jax.random.normal(k_Y, shape=(n, d))
    
    R_base = jax.random.bernoulli(k_wb, p=0.5, shape=(B + 1, n)) * 2.0 - 1.0
    R_base = R_base.at[0].set(jnp.ones(n))
    R_half = R_base.T 
    R = jnp.concatenate((R_half, -R_half), axis=0) 
    
    paired_indices = jnp.diag_indices(n)
    wild_mask = jnp.zeros((2 * n, 2 * n), dtype=bool)
    wild_mask = wild_mask.at[paired_indices].set(True)
    wild_mask = wild_mask.at[paired_indices[0] + n, paired_indices[1]].set(True)
    wild_mask = wild_mask.at[paired_indices[0], paired_indices[1] + n].set(True)
    wild_mask = wild_mask.at[paired_indices[0] + n, paired_indices[1] + n].set(True)
    
    def compute_mmd_u(Y_k):
        Z = jnp.concatenate((X, Y_k), axis=0)
        
        def scan_dim(carry, i):
            Z_i = Z[:, i, None] 
            dist = jnp.abs(Z_i - Z_i.T)
            return carry + dist, None
            
        D_l1, _ = jax.lax.scan(scan_dim, jnp.zeros((2 * n, 2 * n)), jnp.arange(d))
        
        K_mat = jnp.exp(-D_l1 / bandwidth)
        K_mat = jnp.where(wild_mask, 0.0, K_mat)
        
        M_k = jnp.sum(R * (K_mat @ R), axis=0) / (n * (n - 1))
        return M_k

    def f(y, z):
        return jnp.sqrt(1.0 - epsilon**2) * y + epsilon * z

    M_1 = compute_mmd_u(Y_1)
    
    keys_z = jax.random.split(k_seq, K - 1)
    k_indices = jnp.arange(2, K + 1)
    
    def scan_sequence(carry, z_key_and_k):
        Y_prev = carry
        z_key, k_idx = z_key_and_k
        
        Z_noise = jax.random.normal(z_key, shape=(n, d))
        
        input_y = jnp.where(k_idx == 3, Y_prev / sigma, Y_prev)
        base_val = f(input_y, Z_noise)
        Y_curr = jnp.where(k_idx == 2, sigma * base_val, base_val)
        
        M_k = compute_mmd_u(Y_curr)
        return Y_curr, M_k
        
    _, M_rest = jax.lax.scan(scan_sequence, Y_1, (keys_z, k_indices))
    
    M_matrix = jnp.concatenate([M_1[None, :], M_rest], axis=0)
    return M_matrix.T


def all_seq_test(key, n, d, epsilon, sigma, K, B, bandwidth, alpha=0.05, break_ties=True):
    """
    Evaluates the Sequential SB test, SB Min, Worst-Case Min, 
    and an Oracle permutation test strictly on Y_2.
    """
    # use the same key for cols and oracle
    keys = jax.random.split(key, 4)
    k_gen, k_seq, k_cols, k_min = keys
    k_oracle = k_cols

    # generate statistics matrix 
    stats_matrix = generate_seq_data_matrix(k_gen, n, d, epsilon, sigma, K, B, bandwidth)  # (B+1, K)
    
    # sequential SB test
    p_val_sb_min_seq, seq_stage = seq_sb_test(stats_matrix, alpha=alpha, key=k_seq)

    B_plus_1 = B + 1

    # SB Min & Worst-Case Min
    
    # rank columns (larger statistic -> smaller p-value rank)
    if break_ties:
        U_cols = jax.random.uniform(k_cols, shape=stats_matrix.shape)
        sort_indices = jnp.lexsort((U_cols, -stats_matrix), axis=0)
    else:
        sort_indices = jnp.argsort(-stats_matrix, axis=0)
        
    ranks = jnp.argsort(sort_indices, axis=0)
    p_vals_matrix = (ranks + 1.0) / B_plus_1

    # calculate row-wise minimums across the K stages
    row_min = jnp.min(p_vals_matrix, axis=1)

    # theoretical Worst-Case Min (Bonferroni)
    p_val_worst_case_min = jnp.minimum(1.0, K * row_min[0])

    # Data-Driven SB Min (Calibrated via permutation ranks)
    x_0_min = row_min[0]
    if break_ties:
        U_min = jax.random.uniform(k_min, shape=row_min.shape)
        u_0_min = U_min[0]
        
        # smaller row_min indicates higher significance
        strict_less_min = row_min < x_0_min
        tie_breaker_min = (row_min == x_0_min) & (U_min <= u_0_min)
        rank_min = jnp.sum(strict_less_min | tie_breaker_min)
    else:
        rank_min = jnp.sum(row_min <= x_0_min)

    p_val_sb_min = rank_min / B_plus_1

    # Oracle Single Test on Y_2
    
    # Y_2 corresponds to k=2, which is index 1 in the K stages
    stat_Y2 = stats_matrix[:, 1]
    
    # larger statistic indicates higher significance
    X_flat = -stat_Y2
    x_0_oracle = X_flat[0]

    if break_ties:
        U_oracle = jax.random.uniform(k_oracle, shape=X_flat.shape)
        u_0_oracle = U_oracle[0]
        
        strict_less_oracle = X_flat < x_0_oracle
        tie_breaker_oracle = (X_flat == x_0_oracle) & (U_oracle <= u_0_oracle)
        rank_oracle = jnp.sum(strict_less_oracle | tie_breaker_oracle)
    else:
        rank_oracle = jnp.sum(X_flat <= x_0_oracle)

    p_val_oracle_single = rank_oracle / B_plus_1

    return p_val_sb_min_seq, p_val_sb_min, p_val_worst_case_min, p_val_oracle_single

@partial(jax.jit, static_argnames=['B', 'alpha', 'B3'])
def mmdagg_all(key, data, B=499, alpha=0.05, B3=50):
    """
    Computes MMD Aggregated test rejections using:
    1. MaxT (Bisection)
    2. SB (No Tie-breaking)
    3. TB (No Tie-breaking)
    """
    X, Y = data
    m = X.shape[0]
    n = Y.shape[0]
    m_plus_n = m + n
    d_dim = X.shape[1]
    
    # bandwidths
    h = min(m, n)
    h_safe = max(h, 3) 
    val = (2.0 / d_dim) * np.log2(h_safe / np.log(np.log(h_safe)))
    J = max(1, int(np.ceil(val)))
    bandwidths = 2.0 ** -np.arange(1, J + 1)
    
    # permutation
    Z = jnp.concatenate([X, Y], axis=0)
    D2 = jnp.sum((Z[:, None, :] - Z[None, :, :])**2, axis=-1)
    
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, (2 * B) + 1)
    idx = jax.vmap(jax.random.permutation, in_axes=(0, None))(keys, jnp.arange(m_plus_n))
    
    v_x = jnp.concatenate((jnp.ones(m), jnp.zeros(n)))
    v_y = jnp.concatenate((jnp.zeros(m), jnp.ones(n)))
    V_X = v_x[idx].at[0].set(v_x).T
    V_Y = v_y[idx].at[0].set(v_y).T
    eye_mask = jnp.eye(m_plus_n, dtype=bool)
    
    # MMD U-Statistic
    def scan_bandwidth(carry, bandwidth):
        K = jnp.exp(-D2 / (2.0 * (bandwidth ** 2)))
        K = jnp.where(eye_mask, 0.0, K)
        
        K_VX = K @ V_X
        K_VY = K @ V_Y
        
        sum_XX = jnp.sum(V_X * K_VX, 0)
        sum_YY = jnp.sum(V_Y * K_VY, 0)
        sum_XY = jnp.sum(V_X * K_VY, 0)
        
        M_k_i = (sum_XX / (m * (m - 1))) + (sum_YY / (n * (n - 1))) - (2.0 * sum_XY / (m * n))
        return None, M_k_i

    _, M = jax.lax.scan(scan_bandwidth, None, bandwidths)
    
    K_dim = M.shape[0]
    M_original = M[:, 0]
    M1 = M[:, :B + 1]              
    M2 = M[:, B + 1:]              
    M1_sorted = jnp.sort(M1, axis=1)  

    # bisection method for MaxT
    weights = jnp.ones(K_dim)
    
    def bisection_step(step, state):
        u_min_curr, u_max_curr = state
        u = (u_max_curr + u_min_curr) / 2.0
        
        idx = jnp.clip(jnp.ceil((B + 1.0) * (1.0 - u * weights)).astype(jnp.int32) - 1, 0, B)
        q = jnp.take_along_axis(M1_sorted, idx[:, None], axis=1).squeeze(1)
        
        P_u = jnp.sum(jnp.max(M2 - q[:, None], axis=0) > 0) / B
        return jax.lax.cond(P_u <= alpha, lambda: (u, u_max_curr), lambda: (u_min_curr, u))

    u_final, _ = jax.lax.fori_loop(0, B3, bisection_step, (0.0, jnp.min(1.0 / weights)))
    
    idx_final = jnp.clip(jnp.ceil((B + 1.0) * (1.0 - u_final * weights)).astype(jnp.int32) - 1, 0, B)
    quantiles_bisect = jnp.take_along_axis(M1_sorted, idx_final[:, None], axis=1).squeeze(1)
    reject_maxT_bisect = jnp.any(M_original > quantiles_bisect)

    # WC and Aggegation
    M1_i = M1[:, :, None]
    M1_b = M1[:, None, :]
    p_sb_cons = jnp.sum(M1_i >= M1_b, axis=1) / (B + 1.0)
    X_sb_cons = jnp.min(p_sb_cons, axis=0)
    
    M2_ge_M1 = M2[:, :, None] >= M1_b
    p_tb_cons = (1.0 + jnp.sum(M2_ge_M1, axis=1)) / (B + 1.0) 
    X_tb_cons = jnp.min(p_tb_cons, axis=0)

    # final tests (no tie breaking)
    reject_sb_no_ties = jnp.sum(X_sb_cons <= X_sb_cons[0]) / (B + 1.0) <= alpha
    reject_tb_no_ties = jnp.sum(X_tb_cons <= X_tb_cons[0]) / (B + 1.0) <= alpha
    
    return (
        reject_maxT_bisect,   # 0
        reject_sb_no_ties,    # 1
        reject_tb_no_ties     # 2
    )

@partial(jax.jit, static_argnames=['B', 'alpha', 'B3'])
def hsicagg_all(key, data, B=499, alpha=0.05, B3=50):
    """
    Computes HSIC Aggregated test rejections using:
    1. MaxT (Bisection)
    2. SB (No Tie-breaking)
    3. TB (No Tie-breaking)
    """
    X, Y = data
    n = X.shape[0]
    d_X = X.shape[1]
    d_Y = Y.shape[1]
    d_dim = d_X + d_Y
    
    # bandwidths (shared for X and Y as Sobolev assumption)
    h = n
    h_safe = max(h, 3) 
    val = (2.0 / d_dim) * np.log2(h_safe / np.log(np.log(h_safe)))
    J = max(1, int(np.ceil(val)))
    bandwidths = 2.0 ** -np.arange(1, J + 1)
    
    # permutation
    D2_X = jnp.sum((X[:, None, :] - X[None, :, :])**2, axis=-1)
    D2_Y = jnp.sum((Y[:, None, :] - Y[None, :, :])**2, axis=-1)
    
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, 2 * B)
    
    # For HSIC, the 0-th index must strictly be the unpermuted identity
    idx_perms = jax.vmap(jax.random.permutation, in_axes=(0, None))(keys, jnp.arange(n))
    idx = jnp.vstack([jnp.arange(n), idx_perms])
    
    eye_mask = jnp.eye(n, dtype=bool)
    
    # HSIC U-Statistic
    def scan_bandwidth(carry, bandwidth):
        # kernel on X
        K = jnp.exp(-D2_X / (2.0 * (bandwidth ** 2)))
        K = jnp.where(eye_mask, 0.0, K)
        
        # kernel on Y
        L = jnp.exp(-D2_Y / (2.0 * (bandwidth ** 2)))
        L = jnp.where(eye_mask, 0.0, L)
        
        # precompute static sums
        K_sum = jnp.sum(K)
        K_row_sums = jnp.sum(K, axis=1)
        
        L_sum = jnp.sum(L)
        L_row_sums = jnp.sum(L, axis=1)
        
        hsic_term_1 = K_sum * L_sum / ((n - 1.0) * (n - 2.0))
        
        def compute_hsic(index):
            # apply permutation strictly to K's rows and columns
            hsic_term_2 = jnp.sum(K[index[:, None], index[None, :]] * L)
            hsic_term_3 = jnp.dot(K_row_sums[index], L_row_sums) / (n - 2.0)
            return (hsic_term_1 + hsic_term_2 - 2.0 * hsic_term_3) / (n * (n - 3.0))

        # vectorize calculation across all 2B+1 index permutations simultaneously
        M_k_i = jax.vmap(compute_hsic)(idx)
        return None, M_k_i

    _, M = jax.lax.scan(scan_bandwidth, None, bandwidths)
    
    K_dim = M.shape[0]
    M_original = M[:, 0]
    M1 = M[:, :B + 1]              
    M2 = M[:, B + 1:]              
    M1_sorted = jnp.sort(M1, axis=1)  

    # bisection method for MaxT test
    weights = jnp.ones(K_dim)
    
    def bisection_step(step, state):
        u_min_curr, u_max_curr = state
        u = (u_max_curr + u_min_curr) / 2.0
        
        idx_b = jnp.clip(jnp.ceil((B + 1.0) * (1.0 - u * weights)).astype(jnp.int32) - 1, 0, B)
        q = jnp.take_along_axis(M1_sorted, idx_b[:, None], axis=1).squeeze(1)
        
        P_u = jnp.sum(jnp.max(M2 - q[:, None], axis=0) > 0) / B
        return jax.lax.cond(P_u <= alpha, lambda: (u, u_max_curr), lambda: (u_min_curr, u))

    u_final, _ = jax.lax.fori_loop(0, B3, bisection_step, (0.0, jnp.min(1.0 / weights)))
    
    idx_final = jnp.clip(jnp.ceil((B + 1.0) * (1.0 - u_final * weights)).astype(jnp.int32) - 1, 0, B)
    quantiles_bisect = jnp.take_along_axis(M1_sorted, idx_final[:, None], axis=1).squeeze(1)
    reject_maxT_bisect = jnp.any(M_original > quantiles_bisect)

    # WC and Aggegation
    M1_i = M1[:, :, None]
    M1_b = M1[:, None, :]
    p_sb_cons = jnp.sum(M1_i >= M1_b, axis=1) / (B + 1.0)
    X_sb_cons = jnp.min(p_sb_cons, axis=0)
    
    M2_ge_M1 = M2[:, :, None] >= M1_b
    p_tb_cons = (1.0 + jnp.sum(M2_ge_M1, axis=1)) / (B + 1.0) 
    X_tb_cons = jnp.min(p_tb_cons, axis=0)

    # final tests (no tie breaking)
    reject_sb_no_ties = jnp.sum(X_sb_cons <= X_sb_cons[0]) / (B + 1.0) <= alpha
    reject_tb_no_ties = jnp.sum(X_tb_cons <= X_tb_cons[0]) / (B + 1.0) <= alpha
    
    return (
        reject_maxT_bisect,
        reject_sb_no_ties, 
        reject_tb_no_ties  
    )