"""Gaussian Mixture Model klaszterezés wrapper."""

from sklearn.mixture import GaussianMixture


def gmm_labels(x_scaled, k, random_state=42, covariance_type="full"):
    """
    GMM klaszterezés előre skálázott adaton.
    
    Args:
        x_scaled: Skálázott feature mátrix (n_samples, n_features)
        k: Komponensek (klaszterek) száma
        random_state: Reprodukálhatóság
        covariance_type: Kovariancia típus ('full', 'tied', 'diag', 'spherical')
    
    Returns:
        labels: Klasztercímkék (n_samples,)
    """
    model = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        random_state=random_state
    )
    model.fit(x_scaled)
    return model.predict(x_scaled)
