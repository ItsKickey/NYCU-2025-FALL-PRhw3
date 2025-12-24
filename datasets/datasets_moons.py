import numpy as np
from sklearn.datasets import make_moons

def load_moons(
    n_samples=300,
    noise=0.05,
    random_state=0
):
    X, y = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state
    )
    return X, y
