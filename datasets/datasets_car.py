import pandas as pd

def load_car():
    path = "data/car.data"
    cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    df = pd.read_csv(path, header=None, names=cols)

    # 將字串類別編碼成整數
    df_encoded = df.apply(lambda col: pd.factorize(col)[0])

    X = df_encoded.iloc[:, :-1].values
    y = df_encoded["class"].values
    return X, y
