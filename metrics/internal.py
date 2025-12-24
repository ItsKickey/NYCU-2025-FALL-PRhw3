import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score

def compute_internal_metrics(X, labels):
    # DBSCAN 可能全部是 noise
    if len(set(labels)) <= 1:
        return {
            "n_clusters": 0,
            "silhouette": None,
            "db_index": None
        }

    return {
        "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
        "silhouette": silhouette_score(X, labels),
        "db_index": davies_bouldin_score(X, labels)
    }
