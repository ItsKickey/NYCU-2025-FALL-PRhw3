import pandas as pd

def load_wine():
    path = "data/wine.data"
    cols = [
        "class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
        "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins",
        "Color intensity", "Hue", "OD280/OD315", "Proline"
    ]
    df = pd.read_csv(path, header=None, names=cols)

    X = df.iloc[:, 1:].values
    y = df["class"].values   # labels: 1, 2, 3
    return X, y
