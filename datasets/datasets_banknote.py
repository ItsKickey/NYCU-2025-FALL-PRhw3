import pandas as pd

def load_banknote():
    data = pd.read_csv("data/data_banknote_authentication.txt", header=None)
    data.columns = ["variance", "skewness", "curtosis", "entropy", "class"]
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y
