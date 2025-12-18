# main.py
import argparse
from sklearn.datasets import make_blobs, make_moons, make_circles
from src.algorithms import run_algorithm
from src.evaluation import evaluate

# -- argumentumok definiálása --
parser = argparse.ArgumentParser(description="Cluster analysis CLI")
parser.add_argument("--algo", type=str, default="kmeans",
                    help="Algorithm name (kmeans, kmedoids, agglo, dbscan, fcm)")
parser.add_argument("--dataset", type=str, default="blobs",
                    help="Dataset name (blobs, moons, circles)")
parser.add_argument("--k", type=int, default=3, help="Number of clusters")
parser.add_argument("--eps", type=float, default=0.5, help="DBSCAN eps")
parser.add_argument("--min_samples", type=int, default=5, help="DBSCAN min_samples")
args = parser.parse_args()

# -- adatok előállítása --
if args.dataset == "blobs":
    X, _ = make_blobs(n_samples=800, centers=args.k, cluster_std=1.2, random_state=42)
elif args.dataset == "moons":
    from sklearn.datasets import make_moons
    X, _ = make_moons(n_samples=800, noise=0.07, random_state=42)
elif args.dataset == "circles":
    from sklearn.datasets import make_circles
    X, _ = make_circles(n_samples=800, noise=0.05, factor=0.5, random_state=42)
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")

# -- futtatás --
print(f"Running {args.algo} on {args.dataset} ...")
out = run_algorithm(args.algo, X, k=args.k, eps=args.eps, min_samples=args.min_samples)
metrics = evaluate(X, out["labels"])

print("✅ Done")
print("Metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v:.3f}")
