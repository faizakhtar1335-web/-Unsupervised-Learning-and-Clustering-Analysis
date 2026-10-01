#!/usr/bin/env python3
"""
Unsupervised Learning and Clustering Analysis
Dataset : Wine Recognition dataset - UCI Machine Learning Repository
          (178 wines, 13 chemical measurements; ships with scikit-learn)
Methods : K-Means, Hierarchical (Ward), Gaussian Mixture, DBSCAN
Extras  : model selection (elbow, silhouette, Davies-Bouldin, Calinski-Harabasz, BIC),
          bootstrap stability, ANOVA feature ranking, PCA / t-SNE visualization,
          external validation against the (hidden) grape cultivar labels.

IMPORTANT: the cultivar labels are NOT used to fit any model. They are used only at
the very end to check how well the unsupervised clusters match reality.

Usage
-----
    python wine_clustering.py                 # saves everything to ./clustering_output
    python wine_clustering.py --out results   # custom output folder
    python wine_clustering.py --k 3           # force the number of clusters
    python wine_clustering.py --show          # also open plot windows
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, normalized_mutual_info_score,
                             silhouette_samples, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
K_RANGE = range(2, 11)
CLUSTER_COLORS = ["#e63946", "#2a9d8f", "#f4a261", "#457b9d", "#8d6cab", "#6c757d"]

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "axes.titleweight": "bold"})


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def save_fig(fig: plt.Figure, path: Path, show: bool = False) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"  [saved] {path}")


def section(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def cluster_palette(k: int) -> dict[int, str]:
    return {i: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(k)}


# --------------------------------------------------------------------------- #
# 1. Data loading and preprocessing
# --------------------------------------------------------------------------- #
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return the 13 features and the hidden cultivar label (used only for validation)."""
    raw = load_wine(as_frame=True)
    X = raw.data.copy()
    y = raw.target.map(dict(enumerate(["Cultivar 1", "Cultivar 2", "Cultivar 3"])))
    return X, y


def inspect_and_preprocess(X: pd.DataFrame, figs: Path, tables: Path,
                           show: bool) -> tuple[np.ndarray, StandardScaler]:
    section("1. DATA INSPECTION AND PREPROCESSING")
    print(f"Shape            : {X.shape[0]} rows x {X.shape[1]} features")
    print(f"Missing values   : {int(X.isna().sum().sum())}")
    print(f"Duplicate rows   : {int(X.duplicated().sum())}")

    desc = X.describe().T
    desc["skew"] = X.skew()
    desc.round(3).to_csv(tables / "descriptive_statistics.csv")
    print("\n", desc[["mean", "std", "min", "max", "skew"]].round(2).to_string())

    # --- scale ranges: why standardisation is essential for distance-based clustering
    ranges = (X.max() - X.min()).sort_values(ascending=False)
    print(f"\nLargest range : {ranges.index[0]} = {ranges.iloc[0]:.1f}")
    print(f"Smallest range: {ranges.index[-1]} = {ranges.iloc[-1]:.3f}")

    # --- outliers (IQR rule) - reported, not removed
    q1, q3 = X.quantile(0.25), X.quantile(0.75)
    iqr = q3 - q1
    out = ((X < q1 - 1.5 * iqr) | (X > q3 + 1.5 * iqr)).sum().sort_values(ascending=False)
    out.to_csv(tables / "outlier_counts.csv", header=["outliers"])
    print(f"\nOutliers (IQR rule), top 5:\n{out.head(5).to_string()}")
    print(f"Rows with at least one outlier: "
          f"{int(((X < q1 - 1.5 * iqr) | (X > q3 + 1.5 * iqr)).any(axis=1).sum())}")

    # --- figure: raw scales vs standardised scales
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xs_df = pd.DataFrame(Xs, columns=X.columns)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.boxplot(data=X, orient="h", color="#a8dadc", fliersize=2, ax=axes[0])
    axes[0].set_xscale("log")
    axes[0].set_title("Before Scaling (log axis)")
    axes[0].set_xlabel("Value (log scale)")
    axes[0].set_ylabel("Feature")
    sns.boxplot(data=Xs_df, orient="h", color="#2a9d8f", fliersize=2, ax=axes[1])
    axes[1].set_title("After Standardisation (z-score)")
    axes[1].set_xlabel("Standardised value")
    axes[1].set_ylabel("")
    fig.suptitle("Figure 1: Effect of Feature Scaling", fontsize=14, fontweight="bold", y=1.02)
    save_fig(fig, figs / "01_scaling_effect.png", show)

    # --- figure: correlation heatmap
    corr = X.corr()
    corr.round(3).to_csv(tables / "correlation_matrix.csv")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, vmin=-1, vmax=1, linewidths=0.4,
                annot_kws={"size": 7}, cbar_kws={"label": "Pearson r"}, ax=ax)
    ax.set_title("Figure 2: Feature Correlation Heatmap")
    save_fig(fig, figs / "02_correlation_heatmap.png", show)
    return Xs, scaler


# --------------------------------------------------------------------------- #
# 2. Choosing the number of clusters
# --------------------------------------------------------------------------- #
def evaluate_k(Xs: np.ndarray, figs: Path, tables: Path, show: bool) -> pd.DataFrame:
    section("2. CHOOSING THE NUMBER OF CLUSTERS (K-Means + GMM)")
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(Xs)
        gmm = GaussianMixture(n_components=k, n_init=5, random_state=RANDOM_STATE).fit(Xs)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(Xs, km.labels_),
                     "davies_bouldin": davies_bouldin_score(Xs, km.labels_),
                     "calinski_harabasz": calinski_harabasz_score(Xs, km.labels_),
                     "gmm_bic": gmm.bic(Xs)})
    res = pd.DataFrame(rows)
    res.round(4).to_csv(tables / "k_selection_metrics.csv", index=False)
    print(res.round(3).to_string(index=False))

    best = {"silhouette": int(res.loc[res.silhouette.idxmax(), "k"]),
            "davies_bouldin": int(res.loc[res.davies_bouldin.idxmin(), "k"]),
            "calinski_harabasz": int(res.loc[res.calinski_harabasz.idxmax(), "k"]),
            "gmm_bic": int(res.loc[res.gmm_bic.idxmin(), "k"])}
    print(f"\nBest k per criterion: {best}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    spec = [("inertia", "Elbow Method (Inertia)", "Inertia (within-cluster SSE)", None),
            ("silhouette", "Silhouette Score (higher is better)", "Silhouette score", "max"),
            ("calinski_harabasz", "Calinski-Harabasz (higher is better)", "CH index", "max"),
            ("davies_bouldin", "Davies-Bouldin (lower is better)", "DB index", "min"),
            ("gmm_bic", "GMM BIC (lower is better)", "BIC", "min")]
    for ax, (col, title, ylabel, best_kind) in zip(axes.ravel(), spec):
        ax.plot(res["k"], res[col], "o-", color="#457b9d", linewidth=2)
        if best_kind:
            idx = res[col].idxmax() if best_kind == "max" else res[col].idxmin()
            ax.plot(res.loc[idx, "k"], res.loc[idx, col], "o", color="#e63946",
                    markersize=12, label=f"Best k = {int(res.loc[idx, 'k'])}")
            ax.legend()
        ax.set_title(title)
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(K_RANGE))
    axes[1, 2].axis("off")
    fig.suptitle("Figure 3: Cluster-Validity Metrics for k = 2 to 10",
                 fontsize=14, fontweight="bold", y=1.01)
    save_fig(fig, figs / "03_k_selection.png", show)
    return res


# --------------------------------------------------------------------------- #
# 3. Dendrogram
# --------------------------------------------------------------------------- #
def plot_dendrogram(Xs: np.ndarray, k: int, figs: Path, show: bool) -> None:
    Z = linkage(Xs, method="ward")
    cut_height = (Z[-k, 2] + Z[-(k - 1), 2]) / 2  # a cut that yields k clusters
    fig, ax = plt.subplots(figsize=(13, 5.5))
    dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=9,
               show_contracted=True, color_threshold=cut_height, ax=ax)
    ax.axhline(cut_height, color="black", linestyle="--", linewidth=1.2,
               label=f"Cut for k = {k}")
    ax.set_title("Figure 4: Ward Hierarchical Clustering Dendrogram (last 30 merges)")
    ax.set_xlabel("Cluster (number of wines in brackets)")
    ax.set_ylabel("Ward linkage distance")
    ax.legend()
    save_fig(fig, figs / "04_dendrogram.png", show)


# --------------------------------------------------------------------------- #
# 4. Comparing algorithms
# --------------------------------------------------------------------------- #
def dbscan_eps_plot(Xs: np.ndarray, min_samples: int, figs: Path, show: bool) -> None:
    d, _ = NearestNeighbors(n_neighbors=min_samples).fit(Xs).kneighbors(Xs)
    kdist = np.sort(d[:, -1])
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(kdist, color="#457b9d", linewidth=2)
    ax.set_title(f"Figure 5: DBSCAN k-Distance Plot (k = {min_samples})")
    ax.set_xlabel("Points sorted by distance")
    ax.set_ylabel(f"Distance to {min_samples}th nearest neighbour")
    save_fig(fig, figs / "05_dbscan_kdistance.png", show)


def compare_algorithms(Xs: np.ndarray, y: pd.Series, k: int, tables: Path,
                       figs: Path, show: bool) -> tuple[pd.DataFrame, dict]:
    section("3. COMPARING CLUSTERING ALGORITHMS")
    labels = {
        "K-Means": KMeans(n_clusters=k, n_init=50, random_state=RANDOM_STATE).fit_predict(Xs),
        "Hierarchical (Ward)": AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xs),
        "Hierarchical (Complete)": AgglomerativeClustering(n_clusters=k, linkage="complete").fit_predict(Xs),
        "Hierarchical (Average)": AgglomerativeClustering(n_clusters=k, linkage="average").fit_predict(Xs),
        "Gaussian Mixture": GaussianMixture(n_components=k, n_init=10,
                                            random_state=RANDOM_STATE).fit_predict(Xs),
    }
    dbscan_eps_plot(Xs, 5, figs, show)
    db = DBSCAN(eps=2.2, min_samples=5).fit(Xs)
    labels["DBSCAN (eps=2.2)"] = db.labels_

    rows = []
    for name, lab in labels.items():
        mask = lab != -1  # DBSCAN noise excluded from internal metrics
        n_clusters = len(set(lab[mask]))
        row = {"algorithm": name, "clusters": n_clusters, "noise_points": int((~mask).sum())}
        if n_clusters >= 2:
            row.update({"silhouette": silhouette_score(Xs[mask], lab[mask]),
                        "davies_bouldin": davies_bouldin_score(Xs[mask], lab[mask]),
                        "calinski_harabasz": calinski_harabasz_score(Xs[mask], lab[mask])})
        row.update({"ARI_vs_cultivar": adjusted_rand_score(y, lab),
                    "NMI_vs_cultivar": normalized_mutual_info_score(y, lab)})
        rows.append(row)
    comp = pd.DataFrame(rows)
    comp.round(4).to_csv(tables / "algorithm_comparison.csv", index=False)
    print(comp.round(3).to_string(index=False))
    return comp, labels


# --------------------------------------------------------------------------- #
# 5. Final model and stability
# --------------------------------------------------------------------------- #
def fit_final_kmeans(Xs: np.ndarray, k: int) -> KMeans:
    section(f"4. FINAL MODEL: K-MEANS WITH k = {k}")
    km = KMeans(n_clusters=k, n_init=50, random_state=RANDOM_STATE).fit(Xs)
    sizes = pd.Series(km.labels_).value_counts().sort_index()
    print("Cluster sizes:\n", sizes.to_string())
    print(f"Inertia    : {km.inertia_:.2f}")
    print(f"Silhouette : {silhouette_score(Xs, km.labels_):.4f}")
    return km


def bootstrap_stability(Xs: np.ndarray, km: KMeans, k: int, tables: Path,
                        n_boot: int = 200) -> np.ndarray:
    """Re-cluster random 80% subsamples and measure agreement (ARI) with the full model."""
    section("5. STABILITY (200 subsamples of 80%)")
    rng = np.random.default_rng(RANDOM_STATE)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(len(Xs), size=int(0.8 * len(Xs)), replace=False)
        sub = KMeans(n_clusters=k, n_init=10, random_state=int(rng.integers(1e6))).fit(Xs[idx])
        scores.append(adjusted_rand_score(km.labels_[idx], sub.labels_))
    scores = np.array(scores)
    print(f"Mean ARI = {scores.mean():.4f}, min = {scores.min():.4f}, "
          f"5th percentile = {np.percentile(scores, 5):.4f}")
    pd.DataFrame({"ari": scores}).to_csv(tables / "bootstrap_stability.csv", index=False)
    return scores


# --------------------------------------------------------------------------- #
# 6. Visualising the clusters
# --------------------------------------------------------------------------- #
def plot_silhouette(Xs: np.ndarray, labels: np.ndarray, k: int, figs: Path, show: bool) -> None:
    sil = silhouette_samples(Xs, labels)
    avg = sil.mean()
    pal = cluster_palette(k)
    fig, ax = plt.subplots(figsize=(8, 6))
    y_lo = 10
    for c in range(k):
        vals = np.sort(sil[labels == c])
        ax.fill_betweenx(np.arange(y_lo, y_lo + len(vals)), 0, vals, color=pal[c], alpha=0.8,
                         label=f"Cluster {c} (n={len(vals)}, mean={vals.mean():.2f})")
        y_lo += len(vals) + 10
    ax.axvline(avg, color="black", linestyle="--", label=f"Average = {avg:.2f}")
    ax.set_title("Figure 7: Silhouette Plot of the K-Means Solution")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Wines (grouped by cluster)")
    ax.set_yticks([])
    ax.legend(loc="lower right", fontsize=9)
    save_fig(fig, figs / "07_silhouette_plot.png", show)


def plot_projections(Xs: np.ndarray, labels: np.ndarray, km: KMeans, k: int,
                     figs: Path, tables: Path, show: bool) -> PCA:
    pal = cluster_palette(k)
    pca = PCA(random_state=RANDOM_STATE).fit(Xs)
    P = pca.transform(Xs)
    evr = pca.explained_variance_ratio_
    centers = pca.transform(km.cluster_centers_)
    print(f"PCA: PC1 = {evr[0]:.1%}, PC2 = {evr[1]:.1%}, PC1+PC2 = {evr[:2].sum():.1%}")

    tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE,
                init="pca", learning_rate="auto").fit_transform(Xs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for c in range(k):
        m = labels == c
        axes[0].scatter(P[m, 0], P[m, 1], s=45, alpha=0.8, color=pal[c], label=f"Cluster {c}",
                        edgecolor="white", linewidth=0.5)
        axes[1].scatter(tsne[m, 0], tsne[m, 1], s=45, alpha=0.8, color=pal[c],
                        label=f"Cluster {c}", edgecolor="white", linewidth=0.5)
    axes[0].scatter(centers[:, 0], centers[:, 1], marker="X", s=260, color="black",
                    edgecolor="white", linewidth=1.5, label="Centroids", zorder=5)
    axes[0].set_title("PCA Projection")
    axes[0].set_xlabel(f"PC1 ({evr[0]:.1%} variance)")
    axes[0].set_ylabel(f"PC2 ({evr[1]:.1%} variance)")
    axes[1].set_title("t-SNE Projection (perplexity = 30)")
    axes[1].set_xlabel("t-SNE dimension 1")
    axes[1].set_ylabel("t-SNE dimension 2")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("Figure 8: K-Means Clusters in Two Dimensions", fontsize=14,
                 fontweight="bold", y=1.02)
    save_fig(fig, figs / "08_cluster_projections.png", show)

    # PCA loadings
    loadings = pd.DataFrame(pca.components_[:2].T, index=pd.read_csv(tables / "correlation_matrix.csv",
                            index_col=0).index, columns=["PC1", "PC2"])
    loadings.round(3).to_csv(tables / "pca_loadings.csv")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(loadings.sort_values("PC1"), annot=True, fmt=".2f", cmap="coolwarm", center=0,
                cbar_kws={"label": "Loading"}, ax=ax)
    ax.set_title("Figure 9: PCA Loadings (what PC1 and PC2 represent)")
    ax.set_ylabel("Feature")
    save_fig(fig, figs / "09_pca_loadings.png", show)
    return pca


def profile_clusters(X: pd.DataFrame, Xs: np.ndarray, labels: np.ndarray, k: int,
                     figs: Path, tables: Path, show: bool) -> pd.DataFrame:
    section("6. CLUSTER PROFILES")
    pal = cluster_palette(k)
    data = X.copy()
    data["cluster"] = labels

    raw_means = data.groupby("cluster").mean().round(3)
    raw_means.T.to_csv(tables / "cluster_means_raw.csv")
    print("Raw feature means per cluster:\n", raw_means.T.round(2).to_string())

    z = pd.DataFrame(Xs, columns=X.columns)
    z["cluster"] = labels
    z_means = z.groupby("cluster").mean()
    z_means.T.round(3).to_csv(tables / "cluster_means_zscore.csv")

    # ANOVA / eta-squared feature ranking
    rows = []
    for col in X.columns:
        groups = [X.loc[labels == c, col] for c in range(k)]
        f, p = stats.f_oneway(*groups)
        grand = X[col].mean()
        ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ss_t = ((X[col] - grand) ** 2).sum()
        rows.append({"feature": col, "F": f, "p_value": p, "eta_squared": ss_b / ss_t})
    rank = pd.DataFrame(rows).sort_values("eta_squared", ascending=False)
    rank.round(5).to_csv(tables / "feature_importance_anova.csv", index=False)
    print("\nFeatures that best separate the clusters (eta-squared):")
    print(rank.head(6).round(4).to_string(index=False))

    # heatmap
    order = rank["feature"].tolist()
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(z_means[order].T, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1.6,
                vmax=1.6, linewidths=0.5, cbar_kws={"label": "Mean z-score"}, ax=ax)
    ax.set_title("Figure 10: Cluster Profile Heatmap (features ranked by separation power)")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Feature")
    save_fig(fig, figs / "10_cluster_heatmap.png", show)

    # radar chart (top 8 features)
    top = order[:8]
    angles = np.linspace(0, 2 * np.pi, len(top), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for c in range(k):
        vals = z_means.loc[c, top].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=pal[c], linewidth=2.5, label=f"Cluster {c}")
        ax.fill(angles, vals, color=pal[c], alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([t.replace("od280/od315_of_diluted_wines", "OD280/OD315") for t in top],
                       fontsize=9)
    ax.set_rlabel_position(22.5)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title("Figure 11: Radar Chart of the Eight Most Discriminating Features\n"
                 "(mean z-score per cluster)", pad=28, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12))
    save_fig(fig, figs / "11_radar_chart.png", show)

    # boxplots of top-4 features by cluster
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, feat in zip(axes.ravel(), order[:4]):
        sns.boxplot(data=data, x="cluster", y=feat, hue="cluster", palette=pal,
                    legend=False, ax=ax)
        ax.set_title(feat)
        ax.set_xlabel("Cluster")
        ax.set_ylabel(feat)
    fig.suptitle("Figure 12: Distribution of the Four Top Features by Cluster",
                 fontsize=14, fontweight="bold", y=1.01)
    save_fig(fig, figs / "12_cluster_boxplots.png", show)
    return rank


# --------------------------------------------------------------------------- #
# 7. External validation
# --------------------------------------------------------------------------- #
def external_validation(y: pd.Series, labels: np.ndarray, k: int, figs: Path,
                        tables: Path, show: bool) -> pd.DataFrame:
    section("7. EXTERNAL VALIDATION (hidden cultivar labels)")
    ct = pd.crosstab(pd.Series(labels, name="Cluster"), y.rename("Cultivar"))
    ct.to_csv(tables / "cluster_vs_cultivar.csv")
    print(ct.to_string())
    purity = ct.max(axis=1).sum() / ct.values.sum()
    print(f"\nPurity = {purity:.4f}")
    print(f"ARI    = {adjusted_rand_score(y, labels):.4f}")
    print(f"NMI    = {normalized_mutual_info_score(y, labels):.4f}")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", linewidths=0.5,
                cbar_kws={"label": "Number of wines"}, ax=ax)
    ax.set_title("Figure 13: K-Means Clusters vs True Grape Cultivar")
    ax.set_xlabel("True cultivar (not used for clustering)")
    ax.set_ylabel("K-Means cluster")
    save_fig(fig, figs / "13_cluster_vs_cultivar.png", show)
    return ct


def plot_algorithm_comparison(comp: pd.DataFrame, figs: Path, show: bool) -> None:
    d = comp.dropna(subset=["silhouette"]).set_index("algorithm")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col, title, color in [(axes[0], "silhouette", "Silhouette (higher is better)", "#2a9d8f"),
                                  (axes[1], "davies_bouldin", "Davies-Bouldin (lower is better)", "#e76f51"),
                                  (axes[2], "ARI_vs_cultivar", "ARI vs True Cultivar (higher is better)", "#457b9d")]:
        d[col].sort_values().plot.barh(color=color, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(col.replace("_", " "))
        ax.set_ylabel("Algorithm")
        for i, v in enumerate(d[col].sort_values()):
            ax.text(v, i, f" {v:.2f}", va="center", fontsize=9)
    fig.suptitle("Figure 6: Algorithm Comparison", fontsize=14, fontweight="bold", y=1.02)
    save_fig(fig, figs / "06_algorithm_comparison.png", show)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="clustering_output", help="output folder")
    ap.add_argument("--k", type=int, default=None, help="force the number of clusters")
    ap.add_argument("--show", action="store_true", help="display plots interactively")
    args = ap.parse_args()
    if not args.show:
        matplotlib.use("Agg")

    out = Path(args.out)
    figs, tables = out / "figures", out / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    X, y = load_data()
    X.to_csv(out / "wine_dataset.csv", index=False)
    Xs, _ = inspect_and_preprocess(X, figs, tables, args.show)

    metrics = evaluate_k(Xs, figs, tables, args.show)
    k = args.k or int(metrics.loc[metrics.silhouette.idxmax(), "k"])
    print(f"\n>>> Selected k = {k}")

    plot_dendrogram(Xs, k, figs, args.show)
    comp, _ = compare_algorithms(Xs, y, k, tables, figs, args.show)
    plot_algorithm_comparison(comp, figs, args.show)

    km = fit_final_kmeans(Xs, k)
    labels = km.labels_
    bootstrap_stability(Xs, km, k, tables)

    section("VISUALIZATIONS")
    plot_silhouette(Xs, labels, k, figs, args.show)
    plot_projections(Xs, labels, km, k, figs, tables, args.show)
    profile_clusters(X, Xs, labels, k, figs, tables, args.show)
    external_validation(y, labels, k, figs, tables, args.show)

    pd.concat([X, pd.Series(labels, name="cluster")], axis=1).to_csv(
        out / "wine_with_clusters.csv", index=False)
    section("DONE")
    print(f"Figures -> {figs.resolve()}\nTables  -> {tables.resolve()}")


if __name__ == "__main__":
    main()
