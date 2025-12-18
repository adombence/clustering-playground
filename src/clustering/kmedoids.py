"""K-Medoids klaszterezés wrapper standard pipeline-nal."""

from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler


def run_kmedoids(X, k, random_state=42):
    """
    K-Medoids klaszterezés skálázással (backward compatibility).
    
    Args:
        X: Feature mátrix (n_samples, n_features)
        k: Klaszterek száma
        random_state: Reprodukálhatóság
    
    Returns:
        labels: Klasztercímkék (n_samples,)
        model: Tanított KMedoids modell
    """
    x_scaled = StandardScaler().fit_transform(X)
    model = KMedoids(n_clusters=k, random_state=random_state)
    labels = model.fit_predict(x_scaled)
    return labels, model