import pandas as pd
import numpy as np


file_path = './dir_get_K/cora'
df = pd.read_csv(file_path, sep="\t")

idx = df['idx']
print(np.array(idx))