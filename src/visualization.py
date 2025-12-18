"""Klaszterezési vizualizációk (scatter + decision boundary)."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def pca_scatter(X, labels, title, show_variance=True):
    """
    Vizualizálja a klaszterezési eredményeket PCA segítségével.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Az input adatok
    labels : array-like, shape (n_samples,)
        A klaszter címkék
    title : str
        A diagram címe
    show_variance : bool, default=True
        Megjelenítse-e a megőrzött variancia %-ot
    """
    xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    xp = pca.fit_transform(xs)
    
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(xp[:, 0], xp[:, 1], s=10, c=labels, cmap="viridis")
    plt.colorbar(sc, shrink=0.8, label="cluster")
    
    if show_variance:
        evr = pca.explained_variance_ratio_
        plt.title(f"{title}\n(captures {evr.sum()*100:.1f}% variance)")
        plt.xlabel(f"PC1 ({evr[0]*100:.1f}%)")
        plt.ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    else:
        plt.title(title)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
    
    plt.tight_layout()
    plt.show()


def decision_boundary_plot(X, model, title, h=0.02, n_components=2):
    """
    Decision boundary vizualizáció PCA-redukált adaton.
    
    Mesh grid alapú: minden pontra predict, majd heatmap.
    ⚠️ FIGYELEM: Számításigényes! Nagy adathalmazon lassú lehet.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Eredeti feature mátrix
    model : sklearn clustering model
        Tanított klaszterező modell (fit() hívva, predict() elérhető)
    title : str
        Ábra címe
    h : float, default=0.02
        Mesh grid felbontás (kisebb = részletesebb, de lassabb)
    n_components : int, default=2
        PCA komponensek száma (2 = 2D plot)
    
    Example
    -------
    >>> from sklearn.cluster import KMeans
    >>> from sklearn.preprocessing import StandardScaler
    >>> x_scaled = StandardScaler().fit_transform(X)
    >>> kmeans = KMeans(n_clusters=3).fit(x_scaled)
    >>> decision_boundary_plot(X, kmeans, "K-Means Decision Boundary")
    """
    # PCA redukció 2D-re
    xs = StandardScaler().fit_transform(X)
    reduced_data = PCA(n_components=n_components, random_state=42).fit_transform(xs)
    
    # Modell újratanítása 2D adaton (ha nem 2D az eredeti)
    if X.shape[1] != n_components:
        model.fit(reduced_data)
    
    # Mesh grid létrehozása
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict minden mesh pontra
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Vizualizáció
    plt.figure(figsize=(8, 6))
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
        alpha=0.6,
    )
    
    # Adatpontok rárajzolása
    plt.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        c=model.labels_,
        cmap="viridis",
        edgecolor="k",
        s=30,
        linewidth=0.5,
    )
    
    # Klaszterközéppontok (ha van cluster_centers_ attribútum, pl. K-Means)
    if hasattr(model, "cluster_centers_"):
        centroids = model.cluster_centers_
        plt.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="X",
            s=200,
            linewidths=3,
            color="red",
            edgecolor="white",
            zorder=10,
            label="Centroids",
        )
        plt.legend()
    
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.show()


def compare_algorithms_grid(X, algorithms, titles, n_cols=2):
    """
    Több algoritmus eredményének összehasonlítása subplot grid-ben.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature mátrix
    algorithms : list of (labels, centroids_or_None) tuples
        Algoritmusok eredményei
    titles : list of str
        Subplot címek
    n_cols : int, default=2
        Oszlopok száma a grid-ben
    
    Example
    -------
    >>> kmeans_labels = kmeans.labels_
    >>> dbscan_labels = dbscan.labels_
    >>> compare_algorithms_grid(
    ...     X,
    ...     [(kmeans_labels, kmeans.cluster_centers_), 
    ...      (dbscan_labels, None)],
    ...     ["K-Means", "DBSCAN"]
    ... )
    """
    n_algos = len(algorithms)
    n_rows = (n_algos + n_cols - 1) // n_cols
    
    xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    xp = pca.fit_transform(xs)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    axes = np.array(axes).flatten() if n_algos > 1 else [axes]
    
    for idx, ((labels, centroids), title) in enumerate(zip(algorithms, titles)):
        ax = axes[idx]
        scatter = ax.scatter(xp[:, 0], xp[:, 1], s=10, c=labels, cmap="viridis")
        
        if centroids is not None:
            # PCA transzformáció a centroidokra (ha vannak)
            # Feltételezzük, hogy a centroids már a redukált (2D) térben van
            ax.scatter(
                centroids[:, 0],
                centroids[:, 1],
                marker="X",
                s=100,
                color="red",
                edgecolor="white",
                linewidth=2,
                zorder=10,
            )
        
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        plt.colorbar(scatter, ax=ax, label="cluster", shrink=0.8)
    
    # Üres subplot-ok elrejtése
    for idx in range(n_algos, len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    plt.show()


def visualize_membership(fcm, X, title="Membership matrix"):
    """
    Vizualizálja a Fuzzy C-means tagsági értékeit.
    Minden klaszterhez külön subplot mutatja, mennyire tartoznak hozzá a pontok.
    
    Parameters
    ----------
    fcm : FCM object
        A betanított FCM modell (fuzzy-c-means)
        fcm.u alakja: (n_samples, n_clusters)
    X : array-like, shape (n_samples, n_features)
        Az input adatok (eredeti, nem skálázott)
    title : str
        A diagram címe
    """
    xs = StandardScaler().fit_transform(X)
    xp = PCA(n_components=2, random_state=42).fit_transform(xs)
    
    _, axes = plt.subplots(1, fcm.n_clusters, figsize=(12, 3))

    # Egységes axes kezelés: mindig listává alakítjuk
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Az fcmeans library u mátrixa: (n_samples, n_clusters)
    # fcm.u[:, i] egy (n_samples,) vektor - az i-edik klaszter tagsági értékei minden pontra
    for i in range(fcm.n_clusters):
        membership = fcm.u[:, i]  # (n_samples,) vektor
        ax = axes[i] if fcm.n_clusters > 1 else axes[0]
        ax.scatter(xp[:, 0], xp[:, 1], s=10, c=membership, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_title(f"Cluster {i} membership")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
