import numpy as np
from sklearn.preprocessing import StandardScaler
from . import get_all_datasets

def load_dataset(
    name,
    scale=True,
    ignore_label=True,
):
    """
    name: dataset name (string)
    scale: whether to standardize features
    ignore_label: for clustering (True) or external eval (False)
    """

    datasets = get_all_datasets()
    X, y = datasets[name]

    # 1. feature scaling（clustering 一定要）
    if scale:
        X = StandardScaler().fit_transform(X)

    # 2. clustering 時不使用 label
    if ignore_label:
        return X

    return X, y
