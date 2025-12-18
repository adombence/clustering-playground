"""Egységes benchmark runner az összes klaszterezési algoritmushoz."""

import os
import itertools
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..datasets import make_all_datasets
from ..evaluation import evaluate, format_metrics
from ..visualization import pca_scatter
from ..clustering import (
    kmeans_labels,
    kmedoids_labels,
    agglomerative_labels,
    dbscan_labels,
    gmm_labels
)


def product_grid(param_grid: dict):
    """Paraméterrács Descartes-szorzata."""
    keys = list(param_grid.keys())
    vals = [param_grid[k] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))


def run_all_benchmarks(
    save_csv=True,
    visualize_best=True,
    results_dir="results/tables",
    include_external_metrics=True
):
    """
    Futtat minden algoritmust minden adatkészleten, paraméterrácson.
    
    Args:
        save_csv: CSV mentése
        visualize_best: Legjobb konfiguráció vizualizálása algoritmusonként/datasetre
        results_dir: Eredmények mentési könyvtára
        include_external_metrics: Külső metrikák (ARI, NMI) számítása ground truth alapján
    
    Returns:
        DataFrame: Összes eredmény
    """
    datasets = make_all_datasets()

    # Algoritmusok és hozzájuk tartozó paraméter grid
    # Testreszabható az igényeknek megfelelően
    registry = {
        "kmeans": {
            "fn": kmeans_labels,
            "grid": {"k": [2, 3, 4, 5, 6]}
        },
        "kmedoids": {
            "fn": kmedoids_labels,
            "grid": {"k": [2, 3, 4, 5, 6]}
        },
        "agglomerative": {
            "fn": agglomerative_labels,
            "grid": {"k": [2, 3, 4, 5, 6], "linkage": ["ward", "complete", "average"]}
        },
        "gmm": {
            "fn": gmm_labels,
            "grid": {"k": [2, 3, 4, 5, 6]}
        },
        "dbscan": {
            "fn": dbscan_labels,
            "grid": {
                "eps": [0.1, 0.2, 0.3, 0.5, 0.7],
                "min_samples": [3, 5, 10]
            }
        },
    }

    all_rows = []
    
    for ds_name, (X, y, k_true) in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} (n={len(X)}, features={X.shape[1]}, true_k={k_true})")
        print(f"{'='*60}")
        
        x_scaled = StandardScaler().fit_transform(X)

        for algo_name, spec in registry.items():
            print(f"\n  Algoritmus: {algo_name}")
            best_row = None
            
            for params in product_grid(spec["grid"]):
                try:
                    labels = spec["fn"](x_scaled, **params)
                    
                    # Értékelés (sample_size=300 nagy adathalmazon gyorsítás)
                    y_for_eval = y if include_external_metrics else None
                    m = evaluate(x_scaled, labels, y_true=y_for_eval, sample_size=300)
                    
                    row = {
                        "dataset": ds_name,
                        "algo": algo_name,
                        **params,
                        **m
                    }
                    all_rows.append(row)
                    
                    # Track legjobb silhouette alapján
                    if m.get("sil") is not None:
                        if best_row is None or m["sil"] > best_row["metrics"]["sil"]:
                            best_row = {
                                "labels": labels,
                                "metrics": m,
                                "params": params
                            }
                    
                    # Progress - formázott metrikák
                    params_str = ", ".join(f"{k}={v}" for k, v in params.items())
                    metrics_str = format_metrics(m, include_external=include_external_metrics)
                    print(f"    {params_str:30s} → {metrics_str}")
                    
                except Exception as e:
                    print(f"    ⚠️ {params} sikertelen: {e}")
                    row = {
                        "dataset": ds_name,
                        "algo": algo_name,
                        **params,
                        "sil": None,
                        "db": None,
                        "ch": None
                    }
                    if include_external_metrics:
                        row["ari"] = None
                        row["nmi"] = None
                        row["homo"] = None
                        row["compl"] = None
                        row["v_meas"] = None
                        row["ami"] = None
                    all_rows.append(row)

            # Vizualizáció (legjobb konfiguráció)
            if visualize_best and best_row:
                metrics_str = format_metrics(best_row["metrics"], include_external=include_external_metrics)
                params_str = ", ".join(f"{k}={v}" for k, v in best_row["params"].items())
                title = f"{ds_name} — {algo_name.upper()} (best)\n{params_str} | {metrics_str}"
                pca_scatter(X, best_row["labels"], title)

    # DataFrame létrehozása és mentése
    df = pd.DataFrame(all_rows)
    
    if save_csv:
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, "benchmark.csv")
        df.to_csv(out_path, index=False)
        print(f"\n{'='*60}")
        print(f"✅ Eredmények mentve: {out_path}")
        print(f"{'='*60}\n")
    
    return df


if __name__ == "__main__":
    # Futtatás standalone módban
    df_results = run_all_benchmarks()
    print("\nTop 5 eredmény silhouette szerint:")
    print(df_results.nlargest(5, "sil")[["dataset", "algo", "k", "sil", "db", "ch"]])
