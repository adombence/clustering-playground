from sklearn.datasets import make_blobs, make_moons, make_circles, load_iris

def make_all_datasets():
    X_blobs, y_blobs = make_blobs(n_samples=800, centers=4, cluster_std=1.2, random_state=42)
    X_moons, y_moons = make_moons(n_samples=800, noise=0.07, random_state=42)
    X_circles, y_circles = make_circles(n_samples=800, noise=0.05, factor=0.5, random_state=42)
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    return {
        "blobs": (X_blobs, y_blobs, 4),
        "moons": (X_moons, y_moons, 2),
        "circles": (X_circles, y_circles, 2),
        "iris": (X_iris, y_iris, 3),
    }