from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    StandardScaler,
)


def build_preprocessor(
    numeric_features: List[str],
    nominal_features: List[str],
    ordinal_features: Optional[List[str]] = None,
    ordinal_categories: Optional[List[List[str]]] = None,
) -> ColumnTransformer:
    """Construct a reusable preprocessing pipeline for segmentation data."""
    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            [
                ("power_transform", PowerTransformer(method="yeo-johnson", standardize=False)),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if nominal_features:
        nominal_pipeline = Pipeline(
            [
                (
                    "onehot",
                    OneHotEncoder(sparse=False, handle_unknown="ignore"),
                )
            ]
        )
        transformers.append(("nominal", nominal_pipeline, nominal_features))

    if ordinal_features:
        ordinal_pipeline = Pipeline(
            [
                (
                    "ordinal",
                    OrdinalEncoder(categories=ordinal_categories, dtype=np.float64),
                )
            ]
        )
        transformers.append(("ordinal", ordinal_pipeline, ordinal_features))

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)


def get_feature_names(
    preprocessor: ColumnTransformer,
    numeric_features: List[str],
    nominal_features: List[str],
    ordinal_features: Optional[List[str]] = None,
) -> List[str]:
    """Return interpretable feature names after ColumnTransformer transformation."""
    feature_names: List[str] = []

    for name, transformer, cols in preprocessor.transformers_:
        if name == "numeric":
            feature_names.extend(cols)
        elif name == "nominal":
            encoder = transformer.named_steps["onehot"]
            feature_names.extend(encoder.get_feature_names_out(cols).tolist())
        elif name == "ordinal":
            feature_names.extend(cols)

    return feature_names


def transform_data(
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
    return_feature_names: bool = True,
) -> Tuple[pd.DataFrame, Optional[List[str]]]:
    """Apply preprocessing to the dataframe and return transformed features."""
    array = preprocessor.fit_transform(df)
    return_feature_names_list = None
    if return_feature_names:
        return_feature_names_list = get_feature_names(
            preprocessor,
            numeric_features=list(df.select_dtypes(include=["number"]).columns),
            nominal_features=[name for name, trans, cols in preprocessor.transformers_ if name == "nominal" for name in cols],
            ordinal_features=[name for name, trans, cols in preprocessor.transformers_ if name == "ordinal" for name in cols],
        )
    return pd.DataFrame(array, columns=return_feature_names_list), return_feature_names_list
