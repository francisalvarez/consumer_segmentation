from __future__ import annotations

import logging
from pathlib import Path

from src.data_loader import load_data, ensure_data_file
from src.preprocessing import build_preprocessor, transform_data
import src.segmentation as segmentation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("consumer_segmentation")


def create_project_structure(root: Path) -> None:
    for folder in ["data", "notebooks", "src", "outputs", "figures"]:
        (root / folder).mkdir(parents=True, exist_ok=True)


def run_segmentation_workflow(root: Path) -> None:
    create_project_structure(root)
    data_path = ensure_data_file()
    logger.info("Loading data from %s", data_path)
    df = load_data(data_path)

    id_columns = ["CustomerID"] if "CustomerID" in df.columns else []
    variable_types = segmentation.detect_variable_types(df, id_columns=id_columns)

    segmentation_variables = ["Age", "Annual_Income_(k$)", "Spending_Score"]
    descriptor_variables = ["Genre"] if "Genre" in df.columns else []
    business_variables: list[str] = []

    report_dir = root / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building summary reports")
    variable_summary = segmentation.build_variable_summary(df, variable_types)
    missingness = segmentation.build_missingness_report(df)
    correlation = segmentation.build_correlation_matrix(df, segmentation_variables)

    variable_summary.to_csv(report_dir / "variable_summary.csv", index=False)
    missingness.to_csv(report_dir / "missingness_report.csv", index=False)
    correlation.to_csv(report_dir / "correlation_matrix.csv")

    logger.info("Running exploratory visualization")
    segmentation.plot_distributions(df, segmentation_variables, figures_dir / "distributions")
    segmentation.plot_pairwise(df, segmentation_variables, figures_dir / "pairplot_segmentation.png")

    preprocessor = build_preprocessor(
        numeric_features=segmentation_variables,
        nominal_features=descriptor_variables,
    )
    features, feature_names = transform_data(preprocessor, df[segmentation_variables + descriptor_variables])
    if feature_names is not None:
        features.columns = feature_names

    logger.info("Evaluating clustering candidates")
    cluster_range = list(range(2, 7))
    kmeans_scores = segmentation.evaluate_kmeans_ranges(features, cluster_range)
    gmm_scores = segmentation.evaluate_gmm_ranges(features, cluster_range)
    kmeans_scores.to_csv(report_dir / "kmeans_evaluation.csv", index=False)
    gmm_scores.to_csv(report_dir / "gmm_evaluation.csv", index=False)

    best_k = int(kmeans_scores.sort_values("silhouette", ascending=False).iloc[0]["n_clusters"])
    logger.info("Best k according to silhouette from KMeans: %s", best_k)

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(features)
    df["segment"] = labels
    df.to_csv(report_dir / "segmented_data.csv", index=False)

    segmentation.plot_pca(df, segmentation_variables, labels, figures_dir / "pca_clusters.png")

    profile = segmentation.compute_cluster_profiles(df, df["segment"], segmentation_variables, descriptor_variables)
    profile.to_csv(report_dir / "segment_profiles.csv", index=False)

    stability = segmentation.build_stability_report(features, best_k)
    stability.to_csv(report_dir / "stability_report.csv", index=False)

    segment_names = segmentation.assign_segment_names(profile)
    name_map = {seg: name for seg, name in segment_names.items()}
    df["segment_name"] = df["segment"].map(name_map)
    df.to_csv(report_dir / "segmented_data_with_names.csv", index=False)

    logger.info("Workflow complete. Results saved to outputs and figures.")


if __name__ == "__main__":
    run_segmentation_workflow(Path(__file__).resolve().parent)
