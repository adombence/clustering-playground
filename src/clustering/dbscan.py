"""DBSCAN klaszterezés wrapper."""

from sklearn.cluster import DBSCAN


def dbscan_labels(x_scaled, eps=0.3, min_samples=5, metric="euclidean"):
    """
    DBSCAN klaszterezés előre skálázott adaton.
    
    Args:
        x_scaled: Skálázott feature mátrix (n_samples, n_features)
        eps: Maximális távolság két pont között egy szomszédságban
        min_samples: Minimális pontok száma egy mag-ponthoz
        metric: Távolság metrika ('euclidean', 'manhattan', stb.)
    
    Returns:
        labels: Klasztercímkék (n_samples,), -1 = zaj/outlier
    """
    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    return model.fit_predict(x_scaled)
