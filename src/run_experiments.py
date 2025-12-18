import pandas as pd
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from evaluation import evaluate
from algorithms import run_algorithm

def run_all():
    # Példa: moons dataset (skálázva)
    X, _ = make_moons(n_samples=800, noise=0.07, random_state=42)
    X = StandardScaler().fit_transform(X)

    results = []
    grid = [
        ("kmeans",   [{"k": k} for k in range(2, 8)]),
        ("kmedoids", [{"k": k} for k in range(2, 8)]),
        ("agglo",    [{"k": k, "link": link} for k in range(2, 8) for link in ["ward","complete","average"]]),
        ("dbscan",   [{"eps": e, "min_samples": m} for e in [0.1,0.3,0.5,0.7,1.0] for m in [3,5]]),
        ("fcm",      [{"k": k, "m": 2.0} for k in range(2, 8)])
    ]

    for alg, params_list in grid:
        for params in params_list:
            try:
                out = run_algorithm(alg, X, **params)
                metrics = evaluate(X, out["labels"])
                results.append({"algorithm": alg, **params, **metrics, "time": out["time"]})
            except RuntimeError as e:
                print(f"[SKIP] {alg} {params} -> {e}")

    df = pd.DataFrame(results)
    df.to_csv("results/tables/summary.csv", index=False)
    print(df.groupby("algorithm")[["sil","db","ch","time"]].mean().sort_values("sil", ascending=False))

if __name__ == "__main__":
    run_all()
