from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def detect_variable_types(df: pd.DataFrame, id_columns: Optional[List[str]] = None) -> Dict[str, List[str]]:
    id_columns = id_columns or []
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    binary = [col for col in categorical if df[col].nunique(dropna=False) == 2]
    ordinal = []
    if id_columns:
        numeric = [col for col in numeric if col not in id_columns]
        categorical = [col for col in categorical if col not in id_columns]
    return {
        "identifier": id_columns,
        "numeric": numeric,
        "categorical": categorical,
        "binary": binary,
        "ordinal": ordinal,
    }


def build_variable_summary(df: pd.DataFrame, variable_types: Dict[str, List[str]]) -> pd.DataFrame:
    records = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_count = int(df[col].nunique(dropna=False))
        missing = int(df[col].isna().sum())
        cardinality = "high" if unique_count > 0.25 * len(df) else "low"
        if df[col].dtype.kind in "bifc":
            skewness = float(df[col].skew())
            variance = float(df[col].var(ddof=0))
            low_variance = variance < 1e-4
        else:
            skewness = np.nan
            variance = np.nan
            low_variance = unique_count <= 1

        records.append(
            {
                "variable": col,
                "dtype": dtype,
                "unique": unique_count,
                "missing": missing,
                "missing_pct": missing / len(df),
                "cardinality": cardinality,
                "skewness": skewness,
                "low_variance": low_variance,
            }
        )
    return pd.DataFrame(records).sort_values("missing", ascending=False)


def build_missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(missing_pct=lambda x: x["missing_count"] / len(df))
        .reset_index()
        .rename(columns={"index": "variable"})
    )


def build_correlation_matrix(df: pd.DataFrame, numeric_features: List[str]) -> pd.DataFrame:
    return df[numeric_features].corr()


def plot_distributions(df: pd.DataFrame, numeric_features: List[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for col in numeric_features:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[col], kde=True, ax=ax, color="#2c7fb8")
        ax.set_title(f"Distribution of {col}")
        fig.tight_layout()
        fig.savefig(output_dir / f"dist_{col}.png", dpi=150)
        plt.close(fig)


def plot_pairwise(df: pd.DataFrame, variables: List[str], output_path: Path) -> None:
    sns.set(style="whitegrid")
    pairplot = sns.pairplot(df[variables], diag_kind="kde", corner=True)
    pairplot.fig.suptitle("Pairwise relationships for segmentation variables", y=1.02)
    pairplot.savefig(output_path, dpi=150)
    plt.close()


def plot_pca(df: pd.DataFrame, features: List[str], labels: Optional[pd.Series], output_path: Path) -> PCA:
    pca = PCA(n_components=2, random_state=42)
    embedded = pca.fit_transform(df[features])
    result = pd.DataFrame(embedded, columns=["pca_1", "pca_2"])
    loadings = pca.components_.T

    fig, ax = plt.subplots(figsize=(8, 6))
    if labels is not None:
        result["cluster"] = labels.astype(str)
        sns.scatterplot(data=result, x="pca_1", y="pca_2", hue="cluster", palette="tab10", ax=ax, s=70, edgecolor="w", linewidth=0.5)
        ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        sns.scatterplot(data=result, x="pca_1", y="pca_2", color="#2c7fb8", ax=ax, s=70, edgecolor="w", linewidth=0.5)

    # Perceptual map arrows for original variables
    scale_x = np.max(np.abs(embedded[:, 0]))
    scale_y = np.max(np.abs(embedded[:, 1]))
    vectors = loadings * np.array([scale_x, scale_y])
    for i, feature in enumerate(features):
        x_vec, y_vec = vectors[i, 0], vectors[i, 1]
        ax.arrow(0, 0, x_vec, y_vec, color="#444444", width=0.005, head_width=0.06, length_includes_head=True)
        ax.text(x_vec * 1.08, y_vec * 1.08, feature, color="#444444", fontsize=10, ha="center", va="center")

    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("PCA perceptual map with segment clusters")
    ax.axhline(0, color="#d3d3d3", linewidth=0.8)
    ax.axvline(0, color="#d3d3d3", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return pca


def evaluate_kmeans_ranges(
    features: pd.DataFrame,
    cluster_range: List[int],
    random_state: int = 42,
) -> pd.DataFrame:
    from sklearn.cluster import KMeans

    records = []
    for k in cluster_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = model.fit_predict(features)
        score = silhouette_score(features, labels)
        records.append(
            {
                "method": "kmeans",
                "n_clusters": k,
                "silhouette": score,
                "calinski_harabasz": calinski_harabasz_score(features, labels),
                "davies_bouldin": davies_bouldin_score(features, labels),
            }
        )
    return pd.DataFrame(records)


def evaluate_gmm_ranges(features: pd.DataFrame, cluster_range: List[int]) -> pd.DataFrame:
    records = []
    for k in cluster_range:
        model = GaussianMixture(n_components=k, random_state=42)
        model.fit(features)
        labels = model.predict(features)
        records.append(
            {
                "method": "gmm",
                "n_clusters": k,
                "silhouette": silhouette_score(features, labels),
                "aic": model.aic(features),
                "bic": model.bic(features),
                "calinski_harabasz": calinski_harabasz_score(features, labels),
                "davies_bouldin": davies_bouldin_score(features, labels),
            }
        )
    return pd.DataFrame(records)


def compute_cluster_profiles(
    df: pd.DataFrame,
    labels: pd.Series,
    segmentation_features: List[str],
    descriptor_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    profile = df.copy()
    profile["segment"] = labels
    agg_funcs = {col: "mean" for col in segmentation_features}
    if descriptor_features:
        agg_funcs.update({col: lambda x: x.mode().iloc[0] if len(x.dropna()) else np.nan for col in descriptor_features})
    return profile.groupby("segment").agg(agg_funcs).reset_index()


def assign_segment_names(profile: pd.DataFrame) -> Dict[int, str]:
    names = {}
    for _, row in profile.iterrows():
        seg = int(row["segment"])
        income = row.get("Annual_Income_(k$)", np.nan)
        score = row.get("Spending_Score", np.nan)
        age = row.get("Age", np.nan)
        if score >= 70 and income >= 65:
            names[seg] = "Premium High-Value Shoppers"
        elif score >= 70 and income < 65:
            names[seg] = "Engaged Value Seekers"
        elif score < 40 and income >= 65:
            names[seg] = "Affluent Low-Engagement"
        elif score < 40 and income < 65:
            names[seg] = "Moderate Low-Spenders"
        else:
            names[seg] = f"Balanced Shoppers {seg}"
    return names


def build_stability_report(
    features: pd.DataFrame,
    n_clusters: int,
    repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    from sklearn.cluster import KMeans

    base = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    base_labels = base.fit_predict(features)
    scores = []
    for seed in range(random_state, random_state + repeats):
        model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
        labels = model.fit_predict(features)
        score = adjusted_rand_score(base_labels, labels)
        scores.append(score)
        logger.info("Bootstrap stability run %s: ARI=%.3f", seed, score)
    return pd.DataFrame({"seed": list(range(random_state, random_state + repeats)), "adjusted_rand": scores})
