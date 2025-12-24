import csv
from datasets.preprocess import load_dataset
from metrics.internal import compute_internal_metrics
from metrics.external import compute_external_metrics
import os
from algorithms import kmeans, agglomerative, dbscan, gmm, spectral
import json
with open("config/sweep_table.json", "r") as f:
    SWEEP_TABLE = json.load(f)


ALGORITHMS = {
    "kmeans": kmeans,
    "agglomerative": agglomerative,
    "dbscan": dbscan,
    "gmm": gmm,
    "spectral": spectral,
}



def run_dataset(dataset_name):
    os.makedirs("results", exist_ok=True)
    toy = True
    if dataset_name in ['Moons','Circles']:
        X = load_dataset(dataset_name, scale=True, ignore_label=True)
    else :
        X, y = load_dataset(dataset_name, scale=True, ignore_label=False)
        toy = False

    for algo_name, algo in ALGORITHMS.items():
        sweep = SWEEP_TABLE[dataset_name][algo_name]

        os.makedirs(f"results/{dataset_name}", exist_ok=True)
        out_path = f"results/{dataset_name}/{dataset_name}_{algo_name}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            titleRow=["dataset", "algorithm",
                "param", "value",
                "n_clusters", "silhouette", "db_index"]
            if not toy:
                titleRow +=["ari","nmi"]

            writer.writerow(titleRow)

            for v in sweep["values"]:
                params = dict(sweep["fixed"])
                params[sweep["param"]] = v

                labels = algo.run(X, params)
                scores = compute_internal_metrics(X, labels)

                dataRow =[dataset_name, algo_name,
                        sweep["param"], v,
                        scores["n_clusters"],
                        scores["silhouette"],
                        scores["db_index"]
                    ]
                if not toy:
                    external = compute_external_metrics(y, labels)
                    dataRow+=[external["ari"],external["nmi"]]
                
                writer.writerow(dataRow)
                



if __name__ == "__main__":
    Lists=['Moons','Circles','Wine','Car','Heart','Banknote']
    for dataset in Lists:
        print(f" Proceeding {dataset}\n")
        run_dataset(dataset)
