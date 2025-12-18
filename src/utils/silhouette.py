
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..evaluation import evaluate


def silhouette_curve(X, clusterer, param_name, values, scale=True, title=None, extra_params=None):
    """
    Silhouette vs egy paraméter (pl. 'k') bármely algoritmusra.

    Args:
        X: nyers feature mátrix
        clusterer: callable(X_scaled, **params) -> labels
        param_name: str, pl. 'k' vagy 'eps'
        values: iterable paraméter értékek
        scale: bool, skálázzuk-e X-et StandardScaler-rel
        title: opcionális cím
        extra_params: dict, további fix paraméterek a clusterer-hez
    
    Returns:
        DataFrame [param_name, silhouette]
    """
    xs = StandardScaler().fit_transform(X) if scale else X
    rows = []

    for v in values:
        params = dict(extra_params or {})
        params[param_name] = v
        try:
            labels = clusterer(xs, **params)
            m = evaluate(xs, labels)
            rows.append((v, m.get("sil", None)))
        except Exception as e:
            print(f"⚠️ {param_name}={v} sikertelen: {e}")
            rows.append((v, None))

    df = pd.DataFrame(rows, columns=[param_name, "silhouette"])

    # Plot
    df_valid = df.dropna()
    if not df_valid.empty:
        plt.figure(figsize=(6, 3.5))
        plt.plot(df_valid[param_name], df_valid["silhouette"], marker="o")
        plt.title(title or f"Silhouette vs {param_name}")
        plt.xlabel(param_name)
        plt.ylabel("silhouette")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    return df

def silhouette_grid(X, clusterer, param_grid, scale=True, title=None):
    """
    Silhouette tetszőleges paraméterrácson (pl. DBSCAN: {'eps': [...], 'min_samples': [...]})
    
    Args:
        X: nyers feature mátrix
        clusterer: callable(X_scaled, **params) -> labels
        param_grid: dict[str, list]
        scale: bool
        title: opcionális cím
    
    Returns:
        DataFrame: minden kombináció + silhouette
    """
    import itertools
    xs = StandardScaler().fit_transform(X) if scale else X
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    rows = []
    for v in itertools.product(*values):
        params = dict(zip(keys, v))
        try:
            labels = clusterer(xs, **params)
            m = evaluate(xs, labels)
            rows.append({**params, "silhouette": m.get("sil", None)})
        except Exception as e:
            print(f"⚠️ {params} sikertelen: {e}")
            rows.append({**params, "silhouette": None})
    
    df = pd.DataFrame(rows)
    if title and not df.dropna(subset=["silhouette"]).empty:
        plt.figure(figsize=(6, 3.5))
        # 1D vizualizáció, ha csak egy paraméter van
        if len(keys) == 1:
            k = keys[0]
            df1 = df.dropna(subset=["silhouette"]).sort_values(k)
            plt.plot(df1[k], df1["silhouette"], marker="o")
            plt.xlabel(k)
            plt.title(title)
        else:
            plt.title(title + " (plot kihagyva, több dimenziós grid)")
        plt.ylabel("silhouette")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return df
