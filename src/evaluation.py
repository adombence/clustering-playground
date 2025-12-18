"""Klaszterezési értékelési metrikák - belső és külső."""

from sklearn.metrics import (
    # Belső metrikák (unsupervised)
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    # Külső metrikák (supervised, ground truth szükséges)
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    adjusted_mutual_info_score,
)
import numpy as np


def evaluate(X, labels, y_true=None, sample_size=None):
    """
    Klaszterezési algoritmusok értékelése belső és opcionális külső metrikákkal.
    
    Args:
        X: Feature mátrix (n_samples, n_features) - skálázott adat
        labels: Prediktált klasztercímkék (n_samples,)
        y_true: Valós címkék (opcionális, külső metrikákhoz)
        sample_size: Silhouette mintavételezés (None = teljes, int = minta méret)
                     Nagy adathalmazon (n > 10000) ajánlott: 1000-5000
    
    Returns:
        dict: Metrikák dictionary-je
            Belső (unsupervised):
                - sil: Silhouette Score (-1 → 1, nagyobb = jobb)
                - db: Davies-Bouldin Index (0 → ∞, kisebb = jobb)
                - ch: Calinski-Harabasz Index (0 → ∞, nagyobb = jobb)
            Külső (supervised, ha y_true megadva):
                - ari: Adjusted Rand Index (-1 → 1, 1 = tökéletes)
                - nmi: Normalized Mutual Information (0 → 1)
                - homo: Homogeneity (0 → 1, minden klaszter 1 osztályt tartalmaz)
                - compl: Completeness (0 → 1, minden osztály 1 klaszterben van)
                - v_meas: V-measure (0 → 1, homo és compl harmonikus átlaga)
                - ami: Adjusted Mutual Information (-1 → 1)
    """
    labels = np.asarray(labels)
    k = len(np.unique(labels[labels >= 0]))  # -1 = DBSCAN noise, kihagyjuk
    n = len(labels)
    
    metrics = {"sil": None, "db": None, "ch": None}

    # Belső metrikák (unsupervised)
    try:
        if 1 < k < n:
            # Silhouette: mintavételezés nagy adathalmazon (gyorsítás)
            if sample_size and n > sample_size:
                metrics["sil"] = silhouette_score(
                    X, labels, 
                    metric="euclidean", 
                    sample_size=sample_size,
                    random_state=42
                )
            else:
                metrics["sil"] = silhouette_score(X, labels)
            
            metrics["db"] = davies_bouldin_score(X, labels)
            metrics["ch"] = calinski_harabasz_score(X, labels)
    except Exception as e:
        # Marad None (pl. ha csak 1 klaszter van)
        pass

    # Külső metrikák (supervised, ha van ground truth)
    if y_true is not None and len(y_true) == n:
        try:
            metrics["ari"] = adjusted_rand_score(y_true, labels)
            metrics["nmi"] = normalized_mutual_info_score(y_true, labels)
            metrics["homo"] = homogeneity_score(y_true, labels)
            metrics["compl"] = completeness_score(y_true, labels)
            metrics["v_meas"] = v_measure_score(y_true, labels)
            metrics["ami"] = adjusted_mutual_info_score(y_true, labels)
        except Exception:
            # Ha valamelyik számítás sikertelen, marad None
            pass

    return metrics


def format_metrics(metrics, include_external=True):
    """
    Metrikák formázott string reprezentációja (táblázatos megjelenítéshez).
    
    Args:
        metrics: evaluate() visszatérési értéke (dict)
        include_external: Külső metrikák megjelenítése (ha False, csak sil, db, ch)
    
    Returns:
        str: Formázott metrikák, pl: "sil=0.756, db=0.339, ch=5938, ari=0.997"
    
    Example:
        >>> m = evaluate(X, labels, y_true=y)
        >>> print(format_metrics(m))
        'sil=0.756, db=0.339, ch=5938, ari=0.997, nmi=0.994, homo=0.965'
    """
    parts = []
    
    # Belső metrikák (mindig megjelenítjük, ha nem None)
    if metrics.get("sil") is not None:
        parts.append(f"sil={metrics['sil']:.3f}")
    if metrics.get("db") is not None:
        parts.append(f"db={metrics['db']:.3f}")
    if metrics.get("ch") is not None:
        parts.append(f"ch={metrics['ch']:.0f}")
    
    # Külső metrikák (opcionális)
    if include_external:
        for key in ["ari", "nmi", "homo", "compl", "v_meas", "ami"]:
            if metrics.get(key) is not None:
                parts.append(f"{key}={metrics[key]:.3f}")
    
    return ", ".join(parts) if parts else "N/A"

