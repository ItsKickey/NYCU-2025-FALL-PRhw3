from sklearn.cluster import AgglomerativeClustering

def run(X, params):
    model = AgglomerativeClustering(
        n_clusters=params["n_clusters"],
        linkage=params["linkage"]
    )
    return model.fit_predict(X)
