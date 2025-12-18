"""Agglomerative (Hierarchical) klaszterezés wrapper."""

from sklearn.cluster import AgglomerativeClustering


def agglomerative_labels(x_scaled, k, linkage="ward"):
    """
    Hierarchikus klaszterezés előre skálázott adaton.
    
    Args:
        x_scaled: Skálázott feature mátrix (n_samples, n_features)
        k: Klaszterek száma
        linkage: Linkage kritérium ('ward', 'complete', 'average', 'single')
    
    Returns:
        labels: Klasztercímkék (n_samples,)
    """
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    return model.fit_predict(x_scaled)
