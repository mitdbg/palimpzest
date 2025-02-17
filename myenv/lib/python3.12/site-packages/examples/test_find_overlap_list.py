from ncls import NCLS

import pickle
import pandas as pd
import numpy as np

# starts = np.random.randint(0, int(1e8), int(1e3))
starts = np.array(range(100))
ends = starts + 100
ids = starts

ncls = NCLS(starts, ends, ids)

starts2 = np.array([0, 10, 20, 40000], dtype=np.int)
ends2 = np.array([5, 15, 25, 50000], dtype=np.int)
indexes2 = np.array([0, 1, 2, 3], dtype=np.int)


print(starts)
print(ncls.has_overlaps(starts2, ends2, indexes2))

# for i in range(0, 100):
#     for j in ncls.find_overlap_list(i, i + 10):
#         print(j)
