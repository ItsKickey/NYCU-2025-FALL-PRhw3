from sklearn.cluster import KMeans

def run(X, params):
    model = KMeans(
        n_clusters=params["n_clusters"],
        init="k-means++",
        random_state=0
    )
    return model.fit_predict(X)
