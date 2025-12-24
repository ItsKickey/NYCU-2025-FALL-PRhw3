from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def compute_external_metrics(y_true, labels):
    return {
        "ari": adjusted_rand_score(y_true, labels),
        "nmi": normalized_mutual_info_score(y_true, labels)
    }
