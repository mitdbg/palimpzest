from palimpzest.corelib import SourceRecord
import numpy as np


def create_sample_matrix(records: list, physical_ops: list, rank: int):
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
        # because the DataSource.getItem method will swap out the input record with a newly
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
