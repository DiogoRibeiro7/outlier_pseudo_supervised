"""Internal stateful missing-value handling for the PSOD estimator."""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer

logger = logging.getLogger(__name__)

Imputer = Union[SimpleImputer, KNNImputer]


def handle_missing_values(
    df: pd.DataFrame,
    *,
    strategy: Optional[str],
    is_training: bool,
    cat_columns: Optional[List[str]],
    imputer: Optional[Imputer],
) -> Tuple[pd.DataFrame, Optional[Imputer]]:
    """Handle missing values and return the updated fitted numeric imputer."""
    if strategy is None:
        return df, imputer

    numeric_cols_all = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols_all) > 0:
        df = df.copy()
        df[numeric_cols_all] = df[numeric_cols_all].replace([np.inf, -np.inf], np.nan)

    if not df.isnull().any().any():
        logger.debug("No missing values detected.")
        return df, imputer

    missing_count = df.isnull().sum().sum()
    logger.info("Handling %s missing values using strategy: %s", missing_count, strategy)

    if strategy == "drop":
        clean = df.dropna()
        dropped_count = len(df) - len(clean)
        if dropped_count > 0:
            logger.warning(
                "Dropped %s rows (%.2f%%) due to missing values.",
                dropped_count,
                dropped_count / len(df) * 100,
            )
        return clean, imputer

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in (cat_columns or []) if col in df.columns]
    numeric_cols = [col for col in numeric_cols if col not in categorical_cols]

    imputed = df.copy()

    if numeric_cols and df[numeric_cols].isnull().any().any():
        if is_training:
            if strategy == "mean":
                imputer = SimpleImputer(strategy="mean")
            elif strategy == "median":
                imputer = SimpleImputer(strategy="median")
            elif strategy == "mode":
                imputer = SimpleImputer(strategy="most_frequent")
            elif strategy == "knn":
                imputer = KNNImputer(n_neighbors=5)

            imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])  # type: ignore[union-attr]
        elif imputer is not None:
            imputed[numeric_cols] = imputer.transform(df[numeric_cols])
        else:
            logger.warning("Imputer not fitted during training. Using mean imputation.")
            fallback = SimpleImputer(strategy="mean")
            imputed[numeric_cols] = fallback.fit_transform(df[numeric_cols])

    for col in categorical_cols:
        if imputed[col].isnull().any():
            mode = imputed[col].mode()
            mode_value = mode.iloc[0] if not mode.empty else imputed[col].iloc[0]
            imputed[col].fillna(mode_value, inplace=True)
            logger.debug("Filled missing values in %s with mode: %s", col, mode_value)

    logger.info(
        "Successfully handled missing values. Remaining missing: %s",
        imputed.isnull().sum().sum(),
    )
    return imputed, imputer
