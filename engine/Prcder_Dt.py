import sklearn
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def preprocessor(x,y):
    y = y.astype(int)
    le = LabelEncoder()
    y = le.fit_transform(np.asarray(y).ravel())

    orig_cols = x.columns.tolist()
    new_cols = list(range(len(orig_cols)))
    x.columns = new_cols
    x = x.astype(float)
    #train_set = xgb.DMatrix(x, label=y)

    map_idx_to_name = pd.DataFrame({
        "feature_idx": new_cols,
        "original_name": orig_cols
    })
    map_name_to_idx = pd.DataFrame({
        "original_name": orig_cols,
        "feature_idx": new_cols
    })
    return x, y, map_idx_to_name, map_name_to_idx