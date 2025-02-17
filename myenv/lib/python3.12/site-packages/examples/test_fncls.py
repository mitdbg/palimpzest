from ncls import FNCLS
import numpy as np

np.random.seed(0)

import pandas as pd

size = int(1e4)

starts = np.random.randint(0, high=int(1e6), size=size) + np.random.random()
ends = starts + np.random.randint(0, high=1000, size=size)
df = pd.DataFrame(data={"Start": starts, "End": ends})

starts = np.random.randint(0, high=int(1e6), size=size) + np.random.random()
ends = starts + np.random.randint(0, high=1000, size=size)
df2 = pd.DataFrame(data={"Start": starts, "End": ends})

print(df)
print(df2)

from time import time

start = time()
fncls = FNCLS(df.Start.values, df.End.values, df.index.values)
end = time()
print("Time:", end - start)
start = time()
qx, sx = fncls.all_overlaps_both(df2.Start.values, df2.End.values, df2.index.values)
end = time()
print("Time:", end - start)
df2.columns = df2.columns + "_b"
j = pd.concat([df.reindex(sx).reset_index(drop=True), df2.reindex(qx).reset_index(drop=True)], axis=1)

print(j.sort_values("Start"))
