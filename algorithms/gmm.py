from sklearn.mixture import GaussianMixture

def run(X, params):
    model = GaussianMixture(
        n_components=params["n_components"],
        covariance_type=params["covariance_type"],
        random_state=0
    )
    return model.fit_predict(X)
