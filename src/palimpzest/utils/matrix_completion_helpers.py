from typing import Dict

import numpy as np

# import pandas as pd
# import scipy
# import torch
# from numpy.linalg import inv
from sklearn.linear_model import Ridge

from palimpzest.corelib.schemas import SourceRecord

# Used in jupyter
# def create_sample_mask(num_rows: int, num_cols: int, rank: int, seed: int):
#     """
#     Compute the observation matrix which will determine which optimizations (cols)
#     are applied to which records (rows).
#     """
#     # if there are fewer physical operators than the rank + 1 for this operation, then every
#     # operation must execute on every record
#     if num_cols <= rank + 1:
#         return np.ones((num_rows, num_cols))

#     # otherwise, we construct an observation matrix which is guaranteed to
#     # have rank + 1 samples per column and per row
#     sample_matrix = np.zeros((num_rows, num_cols))

#     # construct matrix in a way that guarantees rank + 1 samples per column,
#     # with minimal overlap across rows
#     start_idx = 0
#     for col in range(num_cols):
#         end_idx = (start_idx + rank + 1) % num_rows
#         if end_idx > start_idx:
#             sample_matrix[start_idx:end_idx, col] = 1
#         else:
#             sample_matrix[start_idx:num_rows, col] = 1
#             sample_matrix[0:end_idx, col] = 1
#         start_idx = (end_idx - 1) % num_rows

#     # go row-by-row and add samples until all rows also have rank + 1 samples per row
#     row_sums = np.sum(sample_matrix, axis=1)
#     col = 0
#     for row in range(num_rows):
#         row_sum = row_sums[row]
#         while row_sum < rank + 1:
#             if sample_matrix[row, col] == 0:
#                 sample_matrix[row, col] = 1
#                 row_sum += 1
#             col = (col + 1) % num_cols

#     # finally shuffle the rows and columns of the sample matrix
#     rng = np.random.default_rng(seed=seed)
#     rng.shuffle(sample_matrix, axis=0)

#     return sample_matrix

def create_sample_mask(records: list, physical_ops: list, rank: int):
    """
    Compute the observation matrix which will determine which optimizations (cols)
    are applied to which records (rows).
    """
    # compute the number of rows and columns
    num_rows = len(records)
    num_cols = len(physical_ops)

    # create mappings from (record_id --> matrix row) and (physical_op_id --> matrix col)
    record_to_row_map = {}
    for row_idx, record in enumerate(records):
        # NOTE: for scan records only, we need to use record._source_id instead of record._id
        # because the DataSource.get_item method will swap out the input record with a newly
        # constructed record. Thus, one way to ensure that the first operator after the scan
        # will lookup the correct parent record is to simply use the source
        record_id = record._id if record.schema != SourceRecord else record._source_id
        record_to_row_map[record_id] = row_idx

    phys_op_to_col_map = {}
    for col_idx, physical_op in enumerate(physical_ops):
        phys_op_to_col_map[physical_op.op_id] = col_idx

    # if there are fewer physical operators than the rank + 1 for this operation, then every
    # operation must execute on every record
    if num_cols <= rank + 1:
        return np.ones((num_rows, num_cols)), record_to_row_map, phys_op_to_col_map

    # otherwise, we construct an observation matrix which is guaranteed to
    # have rank + 1 samples per column and per row
    sample_matrix = np.zeros((num_rows, num_cols))

    # construct matrix in a way that guarantees rank + 1 samples per column,
    # with minimal overlap across rows
    start_idx = 0
    for col in range(num_cols):
        end_idx = (start_idx + rank + 1) % num_rows
        if end_idx > start_idx:
            sample_matrix[start_idx:end_idx, col] = 1
        else:
            sample_matrix[start_idx:num_rows, col] = 1
            sample_matrix[0:end_idx, col] = 1
        start_idx = (end_idx - 1) % num_rows

    # go row-by-row and add samples until all rows also have rank + 1 samples per row
    row_sums = np.sum(sample_matrix, axis=1)
    col = 0
    for row in range(num_rows):
        row_sum = row_sums[row]
        while row_sum < rank + 1:
            if sample_matrix[row, col] == 0:
                sample_matrix[row, col] = 1
                row_sum += 1
            col = (col + 1) % num_cols

    return sample_matrix, record_to_row_map, phys_op_to_col_map


def create_sample_mask_from_budget(num_rows: int, num_cols: int, budget: int, seed: int):
    """
    Compute the observation matrix which will determine which optimizations (cols)
    are applied to which records (rows).
    """
    # budget must at least be equal to the number of columns
    assert budget >= num_cols, "sample budget must be greater than or equal to number of optimizations"

    # if the budget is larger than the size of the matrix, return an all ones matrix
    if budget >= num_rows * num_cols:
        return np.ones((num_rows, num_cols))

    # otherwise, we construct an observation matrix which is guaranteed to
    # have at least 1 sample per column
    sample_mask = np.zeros((num_rows, num_cols))

    # fully sample 1 row
    rng = np.random.default_rng(seed=seed)
    full_row_index = rng.choice(np.arange(num_rows), size=1)
    sample_mask[full_row_index, :] = 1
    budget -= num_cols

    # sample remainder of matrix, trying to spread samples across columns and rows
    samples_per_column = np.ones(num_cols)
    samples_per_row = np.zeros(num_rows)
    samples_per_row[full_row_index] = 1
    while budget > 0:
        min_samples_row_order = np.argsort(samples_per_row)
        min_samples_col_order = np.argsort(samples_per_column)

        # it is possible for the row with the fewest samples to have an entry in the column
        # with the fewest samples; we go to the row with the next fewest samples if this is the case;
        # note that we don't need to increment the column index because -- by definition of being
        # the column with the fewest samples -- there must be a row which has a 0 in this column
        # (this is also guaranteed b/c if (budget >= num_rows * num_cols) we simply return np.ones((num_rows, num_cols))
        row_idx = 0
        while sample_mask[min_samples_row_order[row_idx], min_samples_col_order[0]] > 0:
            row_idx += 1

        # sample the entry
        row = min_samples_row_order[row_idx]
        col = min_samples_col_order[0]
        sample_mask[row, col] = 1

        # update state variables
        samples_per_row[row] += 1
        samples_per_column[col] += 1
        budget -= 1

    # finally shuffle the rows of the sample matrix
    rng.shuffle(sample_mask, axis=0)

    return sample_mask


def adaptive_stratified_sampling(
    matrices: Dict[str, np.array],
    num_rows: int,
    num_cols: int,
    sample_budget: int,
    batch_size: int=None,
    p_distr: Dict[str, float]={"cost": 1/3, "time": 1/3, "quality": 1/3},
    seed: int=42,
):
    # ensure that each matrix has float dtype (not int)
    matrices = {metric: matrix.astype(float) for metric, matrix in matrices.items()}

    # initialize rng and sample mask
    rng = np.random.default_rng(seed=seed)
    sample_mask = np.zeros((num_rows, num_cols))

    # throw an exception if the sample_budget is less than the number of columns
    assert sample_budget >= 3 * num_cols, "sample budget smaller than initial budget size!"

    # fully sample three rows at random
    sample_row_indices = rng.choice(np.arange(num_rows), size=3, replace=False)
    sample_mask[sample_row_indices, :] = 1

    # set batch size if not specified
    if batch_size is None:
        batch_size = min(8, int(num_cols / 2))

    # iteratively: complete matrices, compute uncertainty, and pick next samples until sample budget is exhausted
    samples_drawn = np.sum(sample_mask)
    while samples_drawn < sample_budget:
        # compute std. dev. of each column for each matrix
        metric_to_stds = {}
        for metric in ["cost", "time", "quality"]:
            col_stds = np.std(matrices[metric], axis=0, where=sample_mask.astype(bool))
            metric_to_stds[metric] = col_stds

        # compute weighted avg. of col. stds
        col_stds = (
            p_distr["cost"] * metric_to_stds["cost"]
            + p_distr["time"] * metric_to_stds["time"]
            + p_distr["quality"] * metric_to_stds["quality"]
        )

        # compute number of samples to draw
        num_samples_to_draw = min(batch_size, sample_budget - samples_drawn)

        if (col_stds == 0).all():
            # give all columns equal weight
            col_alloc_weights = np.ones(num_cols) / num_samples_to_draw
        else:
            # compute ratio of std relative to sum
            col_alloc_weights = col_stds / np.sum(col_stds)

        # allocate samples to columns based on weight
        col_order = np.argsort(-col_alloc_weights)
        allocated_samples = np.zeros(num_cols)
        for col in col_order:
            ideal_weight_allocation = int(np.ceil(col_alloc_weights[col] * num_samples_to_draw))
            samples_left_to_draw = num_samples_to_draw - np.sum(allocated_samples)
            samples_left_in_col = np.sum(sample_mask[:,col]==0) - allocated_samples[col]
            num_alloc_samples = min(ideal_weight_allocation, samples_left_to_draw, samples_left_in_col)
            allocated_samples[col] += num_alloc_samples

            if np.sum(allocated_samples) == num_samples_to_draw:
                break

        # if we still have samples left-over, allocate them one-at-time to columns w/fewest samples --> most
        while np.sum(allocated_samples) < num_samples_to_draw:
            col_order = np.argsort(np.sum(sample_mask, axis=0))
            for col in col_order:
                allocated_samples[col] += 1
                if np.sum(allocated_samples) == num_samples_to_draw:
                    break

        # for each column with non-zero weight, randomly sample a set of previously unobserved entries
        for col, sample_size in enumerate(allocated_samples):
            if int(sample_size) > 0:
                sampled_rows = rng.choice(np.where(sample_mask[:,col]==0)[0], size=int(sample_size), replace=False)
                sample_mask[sampled_rows, col] = 1

        samples_drawn += num_samples_to_draw

    # sanity check that matrix was properly sampled
    assert int(np.sum(sample_mask)) == int(sample_budget)

    # return final completed matrix
    return sample_mask


def adaptive_mab_sampling(
    matrices: Dict[str, np.array],
    num_rows: int,
    num_cols: int,
    sample_budget: int,
    alpha: float=0.25,
    p_distr: Dict[str, float]={"cost": 1/3, "time": 1/3, "quality": 1/3},
    seed: int=42,
):
    # TODO: if we have constraint values, we can make reward 1.0 if it satisfies constraint and 0.0 if it does not

    # ensure that each matrix has float dtype (not int)
    matrices = {metric: matrix.astype(float) for metric, matrix in matrices.items()}

    # compute initial sample budget and perform sanity check
    init_sample_budget = 3 * num_cols
    assert sample_budget >= init_sample_budget, "sample budget smaller than initial budget size!"

    # initialize sample mask by running every operator on same row
    rng = np.random.default_rng(seed=seed)
    sample_row_indices = rng.choice(np.arange(num_rows), size=3, replace=False)
    sample_mask = np.zeros((num_rows, num_cols))
    sample_mask[sample_row_indices, :] = 1

    # initialize column stats
    ucbs = np.ones(num_cols) * np.inf
    col_num_samples = np.ones(num_cols) * 3
    col_costs = {col: list(matrices["cost"][sample_row_indices, col].flatten()) for col in range(num_cols)}
    col_times = {col: list(matrices["time"][sample_row_indices, col].flatten()) for col in range(num_cols)}
    col_qualities = {col: list(matrices["quality"][sample_row_indices, col].flatten()) for col in range(num_cols)}

    # iteratively: compute upper confidence bounds and sample next entry
    samples_drawn = init_sample_budget
    while samples_drawn < sample_budget:
        # sample random row from max ucb column which still has rows to sample
        cols_not_fully_sampled = col_num_samples < num_rows
        ucb_max = np.max(ucbs[cols_not_fully_sampled])
        max_ucb_cols = np.argwhere((ucbs == ucb_max) & cols_not_fully_sampled)
        max_ucb_col = rng.choice(max_ucb_cols.flatten(), size=1)[0]

        # randomly sample an unsampled row in max_ucb_col
        sampled_row = rng.choice(np.where(sample_mask[:, max_ucb_col] == 0)[0], size=1)[0]
        sample_mask[sampled_row, max_ucb_col] = 1

        # observe matrix entries
        cost = matrices["cost"][sampled_row, max_ucb_col]
        time = matrices["time"][sampled_row, max_ucb_col]
        quality = matrices["quality"][sampled_row, max_ucb_col]

        # increment number of samples drawn
        samples_drawn += 1

        # update col_num_samples and col metrics for sampled column
        col_num_samples[max_ucb_col] += 1
        col_costs[max_ucb_col].append(cost)
        col_times[max_ucb_col].append(time)
        col_qualities[max_ucb_col].append(quality)

        # update ucbs of each column
        for col in range(num_cols):
            n_col = col_num_samples[col]
            if n_col > 1:
                # compute std. dev. of each metric normalized into the range [0,1]
                cost_std = np.std(col_costs[col] / np.max(col_costs[col])) if np.max(col_costs[col]) > 0 else 0.0
                time_std = np.std(col_times[col] / np.max(col_times[col]))
                quality_std = np.std(col_qualities[col])

                # compute weighted reward and update ucb
                weighted_reward = p_distr["cost"] * cost_std + p_distr["time"] * time_std + p_distr["quality"] * quality_std
                ucbs[col] = weighted_reward + alpha * np.sqrt(2 * np.log(samples_drawn) / n_col)

    # sanity check that matrix was properly sampled
    assert int(np.sum(sample_mask)) == int(sample_budget)

    # return final completed matrix
    return sample_mask


#########################################################
#########################################################
#########################################################

class CustomALS(object):
    """
    Predicts using ALS
    Credit to Matt Johnson who posted this code here: https://mattshomepage.com/articles/2018/Jul/01/als/
    """
    
    def __init__(self, rank, n_iter=20, lambda_u=0.001, lambda_v=0.001):
        self.rank = rank
        self.n_iter = n_iter
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v

    def fit(self, R):
        self.R = R.copy()

        # Convert missing entries to 0
        self.R = np.nan_to_num(self.R)

        m, n = R.shape
  
        # Initialize
        self.U = np.random.normal(loc=0., scale=0.01, size=(m, self.rank))
        self.V = np.random.normal(loc=0., scale=0.01, size=(n, self.rank))

        I = np.eye(self.rank)
        # Iu = self.lambda_u * I
        # Iv = self.lambda_v * I

        R_T = self.R.T

        model_u = Ridge(alpha=self.lambda_u, fit_intercept=True)
        model_v = Ridge(alpha=self.lambda_v, fit_intercept=True)

        for _ in range(self.n_iter):
            # NOTE: This can be parallelized
            for i in range(m):
                model_u.fit(X=self.V, y=R_T[:,i])       
                self.U[i,:] = model_u.coef_

            # NOTE: This can be parallelized
            for j in range(n):
                model_v.fit(X=self.U, y=R_T[j,:])        
                self.V[j,:] = model_v.coef_

        return self.U.dot(self.V.T)


def als_complete_matrix(obs_matrix, sample_mask, rank):
    # compute column means and standard deviations
    col_means = np.mean(obs_matrix, axis=0)
    col_stds = np.std(obs_matrix, axis=0)

    # # in some cases we may have zero variance in ALL of our observed sample data;
    # # in this case, the rational way to complete the matrix is to assume it is
    # # rank = 1 and every data point is equal to the sample mean
    # if (col_stds == 0.0).all():
    #     als_completed_matrix = np.zeros((obs_matrix.shape))
    #     als_completed_matrix[:, :] = col_means
    #     return als_completed_matrix

    # if we have some columns with 0 variance, set their true_col_stds entries equal to 1;
    # this will ensure that these entries are not scaled, but still have their mean translation
    zero_variance_cols = (col_stds == 0.0)
    col_stds[zero_variance_cols] = 1.0

    # create scaled version of observation matrix
    scaled_obs_matrix = (obs_matrix - col_means) / col_stds

    # set missing entries to np.nan (expected by CustomALS)
    scaled_obs_matrix[~sample_mask.astype(bool)] = np.nan

    # initialize ALS
    als = CustomALS(rank=rank)
    scaled_completed_matrix = als.fit(scaled_obs_matrix)

    # rescale completed matrix back to original size
    als_completed_matrix = scaled_completed_matrix * col_stds + col_means

    return als_completed_matrix

def complete_matrix(
        matrix: np.array,
        sample_mask: np.array,
        rank: int,
        # sample_budget: int,
        metric: str,
        seed: int,
    ):
    # if sample_mask is all 1's, no need to complete the matrix (it is already complete)
    if (sample_mask == 1.0).all():
        return matrix

    # ensure that matrix has float dtype (not int)
    matrix = matrix.astype(float)

    # complete the matrix
    losses, recon_losses, col_losses = [], [], []
    # completed_matrix, losses, recon_losses, col_losses = sgd_complete_matrix(matrix, sample_mask, rank)
    completed_matrix = als_complete_matrix(matrix, sample_mask, rank)
    # completed_matrix = adaptive_complete_matrix(matrix, sample_mask, sample_budget, seed)

    # fix completed matrix entries to match groundtruth samples where we have them
    completed_matrix[sample_mask.astype(bool)] = matrix[sample_mask.astype(bool)]

    # clamp all matrices to be non-negative
    completed_matrix = np.clip(completed_matrix, 0.0, None)

    # clamp quality matrix to be less than 1.0
    if metric == "quality":
        completed_matrix = np.clip(completed_matrix, 0.0, 1.0)

    return completed_matrix, losses, recon_losses, col_losses

def k_largest_index_argpartition(arr, k):
    """
    Get indices of k largest elements in array arr.
    See: https://stackoverflow.com/questions/43386432/how-to-get-indexes-of-k-maximum-values-from-a-numpy-multidimensional-array
    """
    idx = np.argpartition(-arr.ravel(),k)[:k]
    return np.column_stack(np.unravel_index(idx, arr.shape))

def abacus_complete_matrices(
    matrices: Dict[str, np.array],
    sample_budget: int,
    p_distribution: Dict[str, float],
    abacus_num_iters: int,
    S=16,
    rank=1,
    seed=42,
):
    
    # ensure that each matrix has float dtype (not int)
    matrices = {metric: matrix.astype(float) for metric, matrix in matrices.items()}

    # compute initial sample budget and perform sanity check
    num_rows, num_cols = matrices["cost"].shape
    init_sample_budget = num_cols # num_rows + num_cols + 1
    assert sample_budget >= init_sample_budget, "sample budget smaller than initial budget size!"

    # # create initial sample mask via random sampling
    # sample_mask = np.zeros((num_rows, num_cols))
    # xs, ys = np.where(sample_mask==0)
    # unobserved_entries = list(zip(xs, ys))
    # rng = np.random.default_rng(seed=seed)
    # sampled_entries = rng.choice(unobserved_entries, size=init_sample_budget, replace=False)
    # for row, col in sampled_entries:
    #     sample_mask[row, col] = 1.0

    # create initial sample mask
    sample_mask = np.zeros((num_rows, num_cols))
    rng = np.random.default_rng(seed=seed)
    init_sample_row = rng.choice(np.arange(num_rows), size=1, replace=False)
    sample_mask[init_sample_row, :] = 1.0

    # compute k
    k = int(np.ceil((sample_budget - init_sample_budget) / abacus_num_iters))

    # iteratively: complete matrices, compute uncertainty, and pick next samples until sample budget is exhausted
    samples_drawn = init_sample_budget
    sample_budget_exhausted = False
    while not sample_budget_exhausted:
        # complete matrices
        completed_matrices = {metric: [] for metric in ["cost", "time", "quality"]}
        for metric in ["cost", "time", "quality"]:
            for idx in range(S):
                completed_matrix, _, _, _ = complete_matrix(matrices[metric], sample_mask, rank, metric)
                completed_matrices[metric].append(completed_matrix)

        # compute average of completed matrices
        avg_completed_matrices = {
            metric: np.mean(matrices_lst, axis=0)
            for metric, matrices_lst in completed_matrices.items()
        }

        # compute uncertainty matrices; R = RMSE((1-Obs) * (E - E_hat))
        uncertainty_matrices = {}
        for metric in ["cost", "time", "quality"]:
            completed_mats = completed_matrices[metric]
            avg_mat = avg_completed_matrices[metric]
            uncertainty_matrix = np.sqrt(np.mean(
                [((1 - sample_mask) * (mat - avg_mat))**2 for mat in completed_mats],
                axis=0,
            ))
            uncertainty_matrices[metric] = uncertainty_matrix

        # compute weighted avg. of uncertainty matrices
        uncertainty_matrix = sum([
            p_distribution[metric] * uncertainty_matrices[metric]
            for metric in ["cost", "time", "quality"]
        ])

        # TODO?: handle corner case where matrix is fully/over-sampled
        # sample top-k entries from uncertainty matrix which have not already been sampled;
        # we set the uncertainty of sampled values to negative infinity to prevent them
        # from being picked
        uncertainty_matrix[sample_mask.astype(bool)] = -np.inf
        num_samples_to_draw = min(k, sample_budget - samples_drawn)
        sampled_entries = k_largest_index_argpartition(uncertainty_matrix, num_samples_to_draw)
        for row, col in sampled_entries:
            sample_mask[row, col] = 1.0

        if samples_drawn == sample_budget:
            sample_budget_exhausted = True
        else:
            samples_drawn += num_samples_to_draw

    # sanity check that matrix was properly sampled
    assert np.sum(sample_mask) == sample_budget

    # compute and return average of final completed matrices
    abacus_completed_matrices = {
        metric: np.mean(matrices_lst, axis=0)
        for metric, matrices_lst in completed_matrices.items()
    }

    return abacus_completed_matrices, sample_mask


# def gradient_descent(init, steps, grad, proj=lambda x: x, num_to_keep=None):
#     """Projected gradient descent.
    
#     Parameters
#     ----------
#         initial : array
#             starting point
#         steps : list of floats
#             step size schedule for the algorithm
#         grad : function
#             mapping arrays to arrays of same shape
#         proj : function, optional
#             mapping arrays to arrays of same shape
#         num_to_keep : integer, optional
#             number of points to keep
        
#     Returns
#     -------
#         List of points computed by projected gradient descent. Length of the
#         list is determined by `num_to_keep`.
#     """
#     xs = [init]
#     for step in steps:
#         xs.append(proj(xs[-1] - step * grad(xs[-1])))
#         if num_to_keep:
#             xs = xs[-num_to_keep:]
#     return xs


# def update_right(A, S, X):
#     """Update right factor for matrix completion objective."""
#     m, n = A.shape
#     _, k = X.shape
#     Y = np.zeros((n, k))
#     # For each row, solve a k-dimensional regression problem
#     # only over the nonzero projection entries. Note that the
#     # projection changes the least-squares matrix siX so we
#     # cannot vectorize the outer loop.
#     for i in range(n):
#         si = S[:, i]
#         sia = A[si, i]
#         siX = X[si]
#         Y[i,:] = np.linalg.lstsq(siX, sia, rcond=None)[0]
#     return Y


# def update_left(A, S, Y):
#     return update_right(A.T, S.T, Y)


# def alternating_minimization(left, right, update_left, update_right, num_updates):
#     """Alternating minimization."""
#     iterates = [(left, right)]
#     for _ in range(num_updates):
#         left = update_left(right)
#         right = update_right(left)
#         iterates.append((left, right))
#     return iterates[-1]


# def altmin(A, S, rank, num_updates):
#     """Toy implementation of alternating minimization."""
#     m, n = A.shape
#     X = np.random.normal(0, 1, (m, rank))
#     Y = np.random.normal(0, 1, (n, rank))
#     return alternating_minimization(X, Y, 
#                                     lambda Y: update_left(A, S, Y), 
#                                     lambda X: update_right(A, S, X),
#                                     num_updates)


# def sgd_complete_matrix(true_mat, mat_mask, rank):
#     device = "cpu"
#     num_rows, num_cols = true_mat.shape

#     X = torch.empty((num_rows, rank), requires_grad=True)
#     torch.nn.init.normal_(X)
#     Y = torch.empty((rank, num_cols), requires_grad=True)
#     torch.nn.init.normal_(Y)

#     mse_loss = torch.nn.MSELoss()
#     opt_X = torch.optim.Adam([X], lr=1e-3, weight_decay=1e-5)
#     opt_Y = torch.optim.Adam([Y], lr=1e-3, weight_decay=1e-5)
#     opt_X_scheduler = torch.optim.lr_scheduler.StepLR(opt_X, step_size=1000, gamma=0.9)
#     opt_Y_scheduler = torch.optim.lr_scheduler.StepLR(opt_Y, step_size=1000, gamma=0.9)

#     # create tensors for mat mask and true_matrix
#     mat_mask = torch.tensor(mat_mask, dtype=bool)
#     true_matrix = torch.as_tensor(true_mat, dtype=torch.float32, device=device)

#     # # compute matrix mean and std
#     # true_mean = true_matrix[mat_mask].mean()
#     # true_std = true_matrix[mat_mask].std()

#     # compute column means and std deviations
#     true_col_means = torch.tensor(np.mean(true_mat, axis=0, where=mat_mask), dtype=torch.float32, device=device)
#     true_col_stds = torch.tensor(np.std(true_mat, axis=0, where=mat_mask), dtype=torch.float32, device=device)

#     # in some cases we may have zero variance in ALL of our observed sample data;
#     # in this case, the rationale way to complete the matrix is to assume
#     # it is rank = 1 and every data point is equal to the true_mean
#     if (true_col_stds == 0.0).all():
#         R = torch.zeros((true_matrix.shape))
#         R[:, :] = true_col_means
#         return R.detach().numpy(), [], [], []

#     # if we have some columns with 0 variance:
#     # 1. set their true_col_stds entries equal to 1;
#     #    a. this will ensure that these entries are not scaled, but still have their mean translation
#     # 2. (TURNED OFF) update the mat_mask to include these values
#     #    a. subtraction of the column mean --> that every entry will be 0.0
#     #    b. by adding these entries to the mask, we encourage the factorized matrix to respect these constraints
#     zero_variance_cols = (true_col_stds == 0.0)
#     true_col_stds[zero_variance_cols] = 1.0
#     # mat_mask[:,zero_variance_cols] = True

#     # otherwise, scale the matrix and learn factor matrices
#     scaled_true_matrix = (true_matrix - true_col_means)/true_col_stds

#     # # precompute column means for scaled_true_matrix
#     # scaled_true_masked_matrix = torch.masked.masked_tensor(scaled_true_matrix, mat_mask)

#     # # NOTE: np.nan fills masked values, but this tensor should not have any
#     # scaled_true_col_means = scaled_true_masked_matrix.mean(dim=0).to_tensor(np.nan)
#     losses, recon_losses, col_losses = [], [], []
#     for _ in range(2000):
#         opt_X.zero_grad()
#         opt_Y.zero_grad()

#         # compute matrix reconstruction
#         R = torch.matmul(X,Y)

#         # # compute the frobenius norm-squared
#         # loss = torch.linalg.norm(mat_mask * (R - scaled_true_matrix), ord='fro')**2

#         # loss for reconstructing groundtruth values
#         recon_loss = mse_loss(R[mat_mask], scaled_true_matrix[mat_mask])

#         # loss for sending means far from scaled values (which are all 0.0)
#         # col_loss = mse_loss(R.mean(dim=0), torch.zeros(true_col_means.shape, dtype=torch.float32, device=device))

#         # # compute loss as weighted average of contributions
#         # alpha = 0.5
#         # recon_loss_contribution = (1 - alpha) * recon_loss
#         # col_loss_contribution = alpha * col_loss
#         # loss = recon_loss_contribution + col_loss_contribution
#         # loss = mse_loss(torch.matmul(X,Y)[mat_mask], true_matrix[mat_mask])

#         loss.backward()

#         losses.append(loss.item())
#         recon_losses.append(recon_loss_contribution.item())
#         col_losses.append(col_loss_contribution.item())

#         opt_X.step()
#         opt_Y.step()
#         opt_X_scheduler.step()
#         opt_Y_scheduler.step()
        
#         # with torch.no_grad():
#         #     X[:] = X.clamp_(min=0)
#         #     Y[:] = Y.clamp_(min=0)

#     # compute reconstruction
#     R = torch.matmul(X, Y)

#     # scale back to original mean and variance
#     R_scaled = R * true_col_stds + true_col_means

#     # TODO: Undo this?
#     # # for any columns which had 0 variance, set their reconstructed values equal to that column mean
#     # R_scaled[:, zero_variance_cols] = true_col_means[zero_variance_cols]

#     return R_scaled.detach().numpy(), losses, recon_losses, col_losses



### OG: stratified-sampling single-matrix algo
# def adaptive_complete_matrix(
#     matrix: np.array,
#     sample_budget: int,
#     seed=42,
# ):  
#     # ensure that each matrix has float dtype (not int)
#     matrix = matrix.astype(float)

#     # compute initial sample budget and perform sanity check
#     num_rows, num_cols = matrix.shape
#     init_sample_budget = 3*num_cols
#     assert sample_budget >= init_sample_budget, "sample budget smaller than initial budget size!"

#     # create initial sample mask
#     sample_mask = create_sample_mask_from_budget(num_rows, num_cols, init_sample_budget, seed)

#     # scale matrix columns
#     obs_matrix = matrix.copy()
#     col_means = np.mean(obs_matrix, axis=0, where=sample_mask.astype(bool))
#     col_stds = np.std(obs_matrix, axis=0, where=sample_mask.astype(bool))

#     # if we have some columns with 0 variance, set their true_col_stds entries equal to 1;
#     # this will ensure that these entries are not scaled, but still have their mean translation
#     zero_variance_cols = (col_stds == 0.0)
#     col_stds[zero_variance_cols] = 1.0

#     # create scaled version of observation matrix
#     scaled_obs_matrix = (obs_matrix - col_means) / col_stds

#     # compute k
#     k = int(np.ceil((sample_budget - init_sample_budget) / 10))

#     # iteratively: complete matrices, compute uncertainty, and pick next samples until sample budget is exhausted
#     samples_drawn = init_sample_budget
#     sample_budget_exhausted = False
#     while not sample_budget_exhausted:
#         # compute std of each column
#         col_stds = np.std(obs_matrix, axis=0, where=sample_mask.astype(bool))
#         zero_variance_cols = (col_stds == 0.0)
#         col_stds[zero_variance_cols] = np.min(col_stds) # TODO: error?

#         # compute number of samples to draw
#         num_samples_to_draw = min(k, sample_budget - samples_drawn)

#         if (col_stds == 0).all():
#             # give all columns equal weight
#             col_alloc_weights = np.ones(num_cols) / num_samples_to_draw
#         else:
#             # compute ratio of std relative to sum
#             col_alloc_weights = col_stds / np.sum(col_stds)

#         # allocate samples to columns based on weight
#         idx = 0
#         col_order = np.argsort(-col_alloc_weights)
#         allocated_samples = np.zeros(num_cols)
#         while np.sum(allocated_samples) < num_samples_to_draw:
#             col = col_order[idx % num_cols]
#             ideal_weight_allocation = int(np.ceil(col_alloc_weights[col] * num_samples_to_draw))
#             samples_left_to_draw = num_samples_to_draw - np.sum(allocated_samples)
#             samples_left_in_col = np.sum(sample_mask[:,col]==0) - allocated_samples[col]
#             num_alloc_samples = min(ideal_weight_allocation, samples_left_to_draw, samples_left_in_col)
#             allocated_samples[col] += num_alloc_samples
#             idx += 1

#         assert np.sum(allocated_samples) == num_samples_to_draw

#         # for each column with non-zero weight, randomly sample a set of previously unobserved entries
#         for col, sample_size in enumerate(allocated_samples):
#             if int(sample_size) > 0:
#                 sampled_rows = rng.choice(np.where(sample_mask[:,col]==0)[0], size=int(sample_size), replace=False)
#                 sample_mask[sampled_rows, col] = 1

#         if samples_drawn == sample_budget:
#             sample_budget_exhausted = True
#         else:
#             samples_drawn += num_samples_to_draw

#     # sanity check that matrix was properly sampled
#     assert int(np.sum(sample_mask)) == int(sample_budget)

#     # compute and return col estimates from final completed matrix
#     return np.mean(obs_matrix, axis=0, where=sample_mask.astype(bool)), sample_mask
