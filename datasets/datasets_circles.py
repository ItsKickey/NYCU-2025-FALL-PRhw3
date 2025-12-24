from sklearn.datasets import make_circles

def load_circles(
    n_samples=300,
    noise=0.05,
    factor=0.5,
    random_state=0
):
    X, y = make_circles(
        n_samples=n_samples,
        noise=noise,
        factor=factor,
        random_state=random_state
    )
    return X, y
