"""K-Means clustering wrapper"""
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run_kmeans(X, k, random_state=42):
    """
    K-Means klaszterezés skálázással (backward compatibility).

    Args:
        X: Feature mátrix (n_samples, n_features)
        k: Klaszterek száma
        random_state: Reprodukálhatóság

    Returns:
        labels: Klaszter címkék (n_samples,)
        model: Tanított KMeans modell
    """
    X_scaled = StandardScaler().fit_transform(X)
    model = KMeans(n_clusters=k, n_init='auto', random_state=random_state)
    labels = model.fit_predict(X_scaled)
    return labels, model