from sklearn.cluster import DBSCAN

def run(X, params):
    model = DBSCAN(
        eps=params["eps"],
        min_samples=params.get("min_samples", 5)
    )
    return model.fit_predict(X)
