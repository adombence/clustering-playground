"""
Clustering Playground - Klaszterezési algoritmusok összehasonlítása
"""

from .algorithms import run_algorithm, estimate_eps
from .evaluation import evaluate, format_metrics
from .datasets import make_all_datasets
from .visualization import (
    pca_scatter, 
    visualize_membership, 
    decision_boundary_plot, 
    compare_algorithms_grid
)

# Új refactored exportok
from .clustering import (
    kmeans_labels,
    kmedoids_labels,
    agglomerative_labels,
    dbscan_labels,
    gmm_labels,
    run_kmeans,
    run_kmedoids
)
from .experiments import run_all_benchmarks

__all__ = [
    # Legacy
    "run_algorithm",
    "estimate_eps",
    "evaluate",
    "format_metrics",
    "make_all_datasets",
    "pca_scatter",
    "visualize_membership",
    "decision_boundary_plot",
    "compare_algorithms_grid",
    # Refactored
    "kmeans_labels",
    "kmedoids_labels",
    "agglomerative_labels",
    "dbscan_labels",
    "gmm_labels",
    "run_kmeans",
    "run_kmedoids",
    "run_all_benchmarks",
]