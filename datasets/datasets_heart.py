import pandas as pd

def load_heart():
    path = "data/processed.cleveland.data"
    cols = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    df = pd.read_csv(path, header=None, names=cols)
    df = df.replace("?", pd.NA).dropna().astype(float)
    df["target"] = (df["target"] > 0).astype(int)
    X = df.drop(columns=["target"]).values
    y = df["target"].values
    return X, y
