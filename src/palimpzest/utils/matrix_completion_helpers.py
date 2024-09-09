import numpy as np


def create_sample_matrix(num_rows: int, num_cols: int, rank: int):
    """
    Compute the observation matrix which will determine which optimizations (cols)
    are applied to which records (rows).
    """
    # if there is a single physical operator for this operation, then every
    # sentinel plan must execute this operation
    if num_cols == 1:
        return np.ones((num_rows, 1))

    # TODO: remove after running low-rank experiment
    return np.ones((num_rows, num_cols))

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
    
    return sample_matrix
