from sklearn.cluster import SpectralClustering

def run(X, params):
    model = SpectralClustering(
        n_clusters=params["n_clusters"],
        n_neighbors=params["n_neighbors"],
        affinity="nearest_neighbors",
        random_state=0
    )
    return model.fit_predict(X)
