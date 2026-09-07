"""
Core PSOD implementation.
"""

import gc

# Add logging throughout the class
import logging
import os
import pickle
import warnings
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import joblib
import numpy as np
import pandas as pd
from category_encoders import BaseNEncoder, TargetEncoder
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

from ._imputation import Imputer, handle_missing_values
from ._input import convert_datetime_columns, to_dataframe, validate_input

logger = logging.getLogger(__name__)


class PSOD(BaseEstimator):
    """
    Pseudo-Supervised Outlier Detection.

    Get outlier predictions using a pseudo-supervised approach by treating
    each feature as a target and using prediction errors as outlier scores.

    Parameters
    ----------
    n_jobs : int, default=-1
        Number of cores used for the regressor. -1 means all cores.
    cat_columns : List[str] or None, default=None
        List specifying column names of categorical features.
    min_cols_chosen : float, default=0.5
        Minimum percentage of columns to be used for each regressor.
    max_cols_chosen : float, default=1.0
        Maximum percentage of columns to be used for each regressor.
    stdevs_to_outlier : float, default=1.96
        Number of standard deviations from the mean prediction error to flag as outlier.
    sample_frac : float, default=1.0
        Fraction of rows each bagging sample shall use.
    correlation_threshold : float, default=0.05
        Filter out all columns with a correlation closer to 0 than the threshold.
    transform_algorithm : str or None, default="logarithmic"
        Algorithm for transforming numerical columns.
        One of ["logarithmic", "yeo-johnson", "quantile", "box-cox", "none"].
    random_seed : int, default=1
        Initial random seed.
    cat_encode_on_sample : bool, default=False
        Apply categorical encoding to bagging sample if True.
    flag_outlier_on : str, default="both ends"
        Specify whether to flag errors on the "low end", "both ends", or "high end".
    base_learner : Type[RegressorMixin], default=LinearRegression
        A regressor class to be used as the base learner.
    cat_encoder : Type[BaseNEncoder], default=TargetEncoder
        A categorical encoder class from category_encoders.
    contamination : float, default=0.1
        Expected proportion of outliers in the dataset.
    min_samples : int, default=1
        Minimum number of samples required for fitting.
    missing_value_strategy : str, default="mean"
        Strategy for handling missing values. Options:
        - "mean": Replace with mean (numeric) or mode (categorical)
        - "median": Replace with median (numeric only)
        - "mode": Replace with most frequent value
        - "knn": Use K-Nearest Neighbors imputation
        - "drop": Drop rows with missing values
        - None: Do not handle missing values
    auto_convert_datetime : bool, default=False
        If True, automatically convert datetime columns to numeric (int64 nanoseconds).

    Attributes
    ----------
    scores_ : pd.Series or None
        Outlier scores after fitting.
    outlier_classes_ : pd.Series or None
        Binary outlier labels after fitting.
    feature_importances_ : pd.DataFrame or None
        Feature importance scores.
    prediction_errors_ : Dict[str, np.ndarray] or None
        Prediction errors for each feature.
    """

    def __init__(
        self,
        n_jobs: int = -1,
        cat_columns: Union[List[str], None] = None,
        min_cols_chosen: float = 0.5,
        max_cols_chosen: float = 1.0,
        stdevs_to_outlier: float = 1.96,
        sample_frac: float = 1.0,
        correlation_threshold: float = 0.05,
        transform_algorithm: Union[str, None] = "logarithmic",
        random_seed: int = 1,
        cat_encode_on_sample: bool = False,
        flag_outlier_on: str = "both ends",
        base_learner: Type[RegressorMixin] = LinearRegression,
        learner_kwargs: Optional[Dict[str, Any]] = None,
        cat_encoder: Type[BaseNEncoder] = TargetEncoder,
        contamination: float = 0.1,
        min_samples: int = 1,
        missing_value_strategy: Union[str, None] = "mean",
        auto_convert_datetime: bool = False,
    ):
        self.cat_columns = cat_columns
        self.cat_encoders: Dict[str, BaseNEncoder] = {}
        self.numeric_encoders: Union[PowerTransformer, QuantileTransformer, None] = None
        self.regressors: Dict[str, RegressorMixin] = {}
        self.n_jobs = n_jobs
        self.scores: Union[pd.Series, None] = None
        self.outlier_classes: Union[pd.Series, None] = None
        self.min_cols_chosen = min_cols_chosen
        self.max_cols_chosen = max_cols_chosen
        # Store integer versions separately (will be set during fit)
        self._min_cols_chosen_int: Optional[int] = None
        self._max_cols_chosen_int: Optional[int] = None
        self.chosen_columns: Dict[str, List[str]] = {}
        self.cols_with_var: List[str] = []
        self.stdevs_to_outlier = stdevs_to_outlier
        self.sample_frac = sample_frac
        self.correlation_threshold = correlation_threshold
        self.transform_algorithm = transform_algorithm
        self.flag_outlier_on = flag_outlier_on
        self.random_seed = random_seed
        self.cat_encode_on_sample = cat_encode_on_sample
        self.random_generator = np.random.default_rng(self.random_seed)
        self.pred_distribution_stats: Dict[str, float] = {}
        self.base_learner = base_learner
        self.learner_kwargs = learner_kwargs or {}
        self.cat_encoder = cat_encoder
        self.contamination = contamination
        self.min_samples = min_samples
        self.missing_value_strategy = missing_value_strategy
        self.auto_convert_datetime = auto_convert_datetime

        # Add feature_importances_ attribute
        self.feature_importances_: Optional[pd.DataFrame] = None

        # Track fitting status properly
        self._is_fitted = False

        # Store prediction errors for feature contribution analysis
        self.prediction_errors_: Optional[Dict[str, np.ndarray]] = None

        # Store feature names for consistency
        self.feature_names_: Optional[List[str]] = None

        # Store training data shape for validation
        self.n_features_in_: Optional[int] = None

        # Store imputer for missing values
        self.imputer_: Optional[Imputer] = None
        self.missing_value_indices_: Optional[pd.Index] = None

        # Validate parameters
        self._validate_params()

        # Log initialization parameters
        logger.info(
            f"Initialized PSOD with parameters: contamination={contamination}, "
            f"transform_algorithm={transform_algorithm}, base_learner={base_learner.__name__}, "
            f"missing_value_strategy={missing_value_strategy}, "
            f"auto_convert_datetime={auto_convert_datetime}"
        )

    def _validate_params(self):
        """Validate input parameters."""
        if not (0 < self.min_cols_chosen <= 1.0):
            raise ValueError("min_cols_chosen must be between 0 and 1.")
        if not (0 < self.max_cols_chosen <= 1.0):
            raise ValueError("max_cols_chosen must be between 0 and 1.")
        if not (0 <= self.correlation_threshold < 1.0):
            raise ValueError("correlation_threshold must be between 0 and 1.")
        if self.min_cols_chosen > self.max_cols_chosen:
            raise ValueError("min_cols_chosen cannot be higher than max_cols_chosen.")
        if self.flag_outlier_on not in ["low end", "both ends", "high end"]:
            raise ValueError('flag_outlier_on must be one of ["low end", "both ends", "high end"].')
        if not (0 < self.contamination < 1):
            raise ValueError("contamination must be between 0 and 1.")
        if self.min_samples < 1:
            raise ValueError("min_samples must be at least 1.")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be 0. Use -1 for all cores or a positive integer.")

        if self.sample_frac > 1.0:
            warnings.warn(
                "sample_frac is set higher than 1.0. This might lead to overfitting. Recommended value is 1.",
                UserWarning,
            )
        if self.min_cols_chosen < 0.3:
            warnings.warn(
                "min_cols_chosen is set to a very low value of less than 0.3. This may reduce performance.",
                UserWarning,
            )
        if self.correlation_threshold > 0.15:
            warnings.warn(
                "correlation_threshold is set higher than 0.15. This may harm performance. Recommended value is 0.05.",
                UserWarning,
            )

        # Add validation for transform_algorithm values
        valid_transforms = [
            "logarithmic",
            "yeo-johnson",
            "quantile",
            "box-cox",
            "none",
            None,
        ]
        if self.transform_algorithm not in valid_transforms:
            raise ValueError(f"transform_algorithm must be one of {valid_transforms}.")

        # Add validation for missing_value_strategy
        valid_missing_strategies = ["mean", "median", "mode", "knn", "drop", None]
        if self.missing_value_strategy not in valid_missing_strategies:
            raise ValueError(f"missing_value_strategy must be one of {valid_missing_strategies}.")
        if not isinstance(self.auto_convert_datetime, bool):
            raise ValueError("auto_convert_datetime must be a boolean.")

        if self.cat_columns is not None:
            if not isinstance(self.cat_columns, list):
                raise ValueError("cat_columns must be a list of column names or None.")
            if not self.cat_columns:
                raise ValueError("cat_columns must be a non-empty list or None.")
            if not all(isinstance(col, str) for col in self.cat_columns):
                raise ValueError("cat_columns must contain only strings.")
            if len(set(self.cat_columns)) != len(self.cat_columns):
                raise ValueError("cat_columns must not contain duplicate names.")

    def _reset_fit_state(self) -> None:
        """Reset all attributes that are set during fitting."""
        self._is_fitted = False
        self.scores = None
        self.outlier_classes = None
        self.feature_importances_ = None
        self.prediction_errors_ = None
        self.feature_names_ = None
        self.n_features_in_ = None
        self.pred_distribution_stats = {}
        self.chosen_columns = {}
        self.regressors = {}
        self.cat_encoders = {}
        self.cols_with_var = []
        self.imputer_ = None
        self.missing_value_indices_ = None
        self._min_cols_chosen_int = None
        self._max_cols_chosen_int = None
        self.numeric_encoders = None
        logger.debug("Reset fit state before training.")

    def _convert_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert datetime columns to numeric if enabled."""
        return convert_datetime_columns(df, enabled=self.auto_convert_datetime)

    def _to_dataframe(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Convert input to DataFrame if necessary (sklearn compatibility)."""
        return to_dataframe(X, feature_names=self.feature_names_)

    def __str__(self):
        """String representation of PSOD object."""
        return f"""
        PSOD Configuration:
        - n_jobs: {self.n_jobs}
        - cat_columns: {self.cat_columns}
        - min_cols_chosen: {self.min_cols_chosen}
        - max_cols_chosen: {self.max_cols_chosen}
        - stdevs_to_outlier: {self.stdevs_to_outlier}
        - sample_frac: {self.sample_frac}
        - correlation_threshold: {self.correlation_threshold}
        - transform_algorithm: {self.transform_algorithm}
        - random_seed: {self.random_seed}
        - cat_encode_on_sample: {self.cat_encode_on_sample}
        - flag_outlier_on: {self.flag_outlier_on}
        - base_learner: {self.base_learner.__name__}
        - cat_encoder: {self.cat_encoder.__name__}
        - contamination: {self.contamination}
        - auto_convert_datetime: {self.auto_convert_datetime}
        - is_fitted: {self._is_fitted}
        """

    # Add __repr__ method for better debugging
    def __repr__(self):
        """Repr method for better debugging."""
        return (
            f"PSOD(n_jobs={self.n_jobs}, contamination={self.contamination}, "
            f"transform_algorithm='{self.transform_algorithm}', "
            f"base_learner={self.base_learner.__name__})"
        )

    def get_params(self, deep=True):
        """Get parameters for this estimator (sklearn compatibility)."""
        return {
            "n_jobs": self.n_jobs,
            "cat_columns": self.cat_columns,
            "min_cols_chosen": self.min_cols_chosen,
            "max_cols_chosen": self.max_cols_chosen,
            "stdevs_to_outlier": self.stdevs_to_outlier,
            "sample_frac": self.sample_frac,
            "correlation_threshold": self.correlation_threshold,
            "transform_algorithm": self.transform_algorithm,
            "random_seed": self.random_seed,
            "cat_encode_on_sample": self.cat_encode_on_sample,
            "flag_outlier_on": self.flag_outlier_on,
            "base_learner": self.base_learner,
            "learner_kwargs": self.learner_kwargs,
            "cat_encoder": self.cat_encoder,
            "contamination": self.contamination,
            "min_samples": self.min_samples,
            "missing_value_strategy": self.missing_value_strategy,
            "auto_convert_datetime": self.auto_convert_datetime,
        }

    def set_params(self, **params):
        """Set parameters for this estimator (sklearn compatibility)."""
        for param, value in params.items():
            setattr(self, param, value)
        self._validate_params()
        return self

    def get_range_cols(self, df: pd.DataFrame):
        """Calculate actual column ranges based on dataframe size."""
        len_cols = len(df.columns) - 1  # Exclude target column
        if len_cols <= 0:
            self._min_cols_chosen_int = 0
            self._max_cols_chosen_int = 0
            return
        # Store converted integer values separately, preserving original fractions
        self._min_cols_chosen_int = max(int(len_cols * self.min_cols_chosen), 1)
        self._max_cols_chosen_int = min(int(len_cols * self.max_cols_chosen), len_cols)

    def chose_random_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Select random columns.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to select columns from.

        Returns
        -------
        List[str]
            List of chosen column names.
        """
        # Handle case where no columns are available
        if len(df.columns) == 0:
            warnings.warn(
                "No columns available for selection. Cannot build regression model.", UserWarning
            )
            return []

        if not self._min_cols_chosen_int or not self._max_cols_chosen_int:
            nb_cols = 0
        elif self._min_cols_chosen_int == self._max_cols_chosen_int:
            nb_cols = self._min_cols_chosen_int
        else:
            nb_cols = self.random_generator.choice(
                np.arange(self._min_cols_chosen_int, self._max_cols_chosen_int + 1),
                1,
                replace=False,
            )[0]

        # Ensure we don't try to select more columns than available
        nb_cols = min(nb_cols, len(df.columns))

        return self.random_generator.choice(df.columns, nb_cols, replace=False).tolist()  # type: ignore[no-any-return]

    def correlation_feature_selection(self, df: pd.DataFrame, target_col: str) -> List[str]:
        """
        Select columns based on correlation with target column.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to select columns from.
        target_col : str
            Target column for correlation computation.

        Returns
        -------
        List[str]
            List of columns with correlation above threshold.
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        return [
            col
            for col in numerical_cols
            if col != target_col and abs(df[col].corr(df[target_col])) > self.correlation_threshold
        ]

    def col_intersection(self, lst1: List[str], lst2: List[str]) -> List[str]:
        """
        Find intersection of two lists.

        Parameters
        ----------
        lst1 : List[str]
            First list.
        lst2 : List[str]
            Second list.

        Returns
        -------
        List[str]
            List of intersecting elements.
        """
        return np.intersect1d(lst1, lst2).tolist()  # type: ignore[no-any-return]

    def make_outlier_classes(self, df_scores: pd.DataFrame, use_trained_stats=True) -> pd.Series:
        """Convert outlier scores to binary labels."""
        if not use_trained_stats:
            mean_score = df_scores["anomaly"].mean()
            std_score = df_scores["anomaly"].std()
            self.pred_distribution_stats = {
                "mean_score": mean_score,
                "std_score": std_score,
            }

        conditions = []
        if self.flag_outlier_on in ["both ends", "low end"]:
            conditions.append(
                df_scores["anomaly"]
                < self.pred_distribution_stats["mean_score"]
                - self.stdevs_to_outlier * self.pred_distribution_stats["std_score"]
            )
        if self.flag_outlier_on in ["both ends", "high end"]:
            conditions.append(
                df_scores["anomaly"]
                > self.pred_distribution_stats["mean_score"]
                + self.stdevs_to_outlier * self.pred_distribution_stats["std_score"]
            )

        df_scores["anomaly_class"] = np.select(conditions, [1] * len(conditions), default=0)
        self.outlier_classes = df_scores["anomaly_class"]
        return df_scores["anomaly_class"]

    def drop_cat_columns(self, df_scores: pd.DataFrame) -> pd.DataFrame:
        """
        Drop categorical columns from the DataFrame and calculate anomaly score.

        Parameters
        ----------
        df_scores : pd.DataFrame
            DataFrame with predictions.

        Returns
        -------
        pd.DataFrame
            DataFrame with anomaly score.
        """
        if isinstance(self.cat_columns, list):
            df_scores["anomaly"] = df_scores.drop(self.cat_columns, axis=1).mean(axis=1)
        else:
            df_scores["anomaly"] = df_scores.mean(axis=1)
        self.scores = df_scores["anomaly"]
        return df_scores

    def fit_transform_numeric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform numeric data based on the specified transformation algorithm.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame.
        """
        if self.transform_algorithm == "logarithmic":
            # Handle negative values by adding offset
            df_min = df.min().min()
            offset = abs(df_min) + 1 if df_min <= 0 else 0
            return np.log1p(df + offset)  # type: ignore[no-any-return]

        elif self.transform_algorithm == "yeo-johnson":
            scaler = PowerTransformer(method="yeo-johnson")
            df_transformed = scaler.fit_transform(df)
            self.numeric_encoders = scaler
            return pd.DataFrame(df_transformed, columns=df.columns, index=df.index)

        elif self.transform_algorithm == "box-cox":
            # Box-Cox requires positive values
            df_min = df.min().min()
            if df_min <= 0:
                warnings.warn("Box-Cox transformation requires positive values. Adding offset.")
                df = df - df_min + 1
            scaler = PowerTransformer(method="box-cox")
            df_transformed = scaler.fit_transform(df)
            self.numeric_encoders = scaler
            return pd.DataFrame(df_transformed, columns=df.columns, index=df.index)

        elif self.transform_algorithm == "quantile":
            scaler = QuantileTransformer(
                output_distribution="normal", random_state=self.random_seed
            )
            df_transformed = scaler.fit_transform(df)
            self.numeric_encoders = scaler
            return pd.DataFrame(df_transformed, columns=df.columns, index=df.index)

        elif self.transform_algorithm in ["none", None]:
            return df

        else:
            raise ValueError(f"Unknown transformation algorithm: {self.transform_algorithm}")

    def transform_numeric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform numeric data based on the fitted transformation algorithm.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame.
        """
        if self.transform_algorithm == "logarithmic":
            # Use same offset as in fitting
            df_min = df.min().min()
            offset = abs(df_min) + 1 if df_min <= 0 else 0
            return np.log1p(df + offset)  # type: ignore[no-any-return]

        elif (
            self.transform_algorithm in ["yeo-johnson", "box-cox", "quantile"]
            and self.numeric_encoders
        ):
            df_transformed = self.numeric_encoders.transform(df)
            return pd.DataFrame(df_transformed, columns=df.columns, index=df.index)

        elif self.transform_algorithm in ["none", None]:
            return df

        else:
            return df

    def remove_zero_variance(self, df: pd.DataFrame) -> List[str]:
        """
        Remove columns with zero variance.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to check for zero variance.

        Returns
        -------
        List[str]
            List of columns with non-zero variance.
        """
        # Use ddof=0 to avoid warnings for single-sample inputs
        var_data = df.var(ddof=0)
        return var_data[var_data != 0].index.to_list()

    def _validate_input(self, df: pd.DataFrame, is_training: bool = True) -> None:
        """Validate input DataFrame against the estimator's current fit state."""
        validate_input(
            df,
            is_training=is_training,
            min_samples=self.min_samples,
            cat_columns=self.cat_columns,
            is_fitted=self._is_fitted,
            feature_names=self.feature_names_,
        )

    def _handle_missing_values(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Handle missing values and retain the fitted numeric imputer."""
        result, self.imputer_ = handle_missing_values(
            df,
            strategy=self.missing_value_strategy,
            is_training=is_training,
            cat_columns=self.cat_columns,
            imputer=self.imputer_,
        )
        return result

    def _calculate_feature_importances(self) -> None:
        """Calculate and store feature importances."""
        if not self._is_fitted:
            logger.warning("Model not fitted. Cannot calculate feature importances.")
            return

        if not self.regressors:
            logger.warning("No regressors available. Cannot calculate feature importances.")
            return

        importances = {}

        # Method 1: Average prediction error contribution
        error_importances: dict = {}
        for target_col, regressor in self.regressors.items():
            feature_cols = self.chosen_columns.get(target_col, [])

            if not feature_cols:
                logger.debug(f"No feature columns for target {target_col}")
                continue

            # Get feature importance from regressor if available
            if hasattr(regressor, "feature_importances_"):
                reg_importance = regressor.feature_importances_
                if len(reg_importance) != len(feature_cols):
                    logger.warning(
                        f"Feature importance length mismatch for {target_col}. "
                        f"Expected {len(feature_cols)}, got {len(reg_importance)}"
                    )
                    continue

                for i, feature in enumerate(feature_cols):
                    if feature not in error_importances:
                        error_importances[feature] = []
                    error_importances[feature].append(reg_importance[i])

            elif hasattr(regressor, "coef_"):
                # For linear models, use absolute coefficients
                reg_importance = np.abs(regressor.coef_)
                # Handle both 1D and 2D coefficient arrays
                if reg_importance.ndim > 1:
                    reg_importance = reg_importance.flatten()

                if len(reg_importance) != len(feature_cols):
                    logger.warning(
                        f"Coefficient length mismatch for {target_col}. "
                        f"Expected {len(feature_cols)}, got {len(reg_importance)}"
                    )
                    continue

                for i, feature in enumerate(feature_cols):
                    if feature not in error_importances:
                        error_importances[feature] = []
                    error_importances[feature].append(reg_importance[i])

        # Average importances across all models
        for feature, importance_list in error_importances.items():
            if importance_list:
                importances[feature] = np.mean(importance_list)
            else:
                importances[feature] = 0.0  # type: ignore[assignment]

        # Method 2: Frequency of selection
        selection_frequency: dict = {}
        total_models = len(self.regressors)

        for feature_cols in self.chosen_columns.values():
            for feature in feature_cols:
                selection_frequency[feature] = selection_frequency.get(feature, 0) + 1

        for feature in selection_frequency:
            selection_frequency[feature] /= total_models

        # Combine both methods
        all_features = set(importances.keys()) | set(selection_frequency.keys())

        if not all_features:
            logger.warning(
                "Could not calculate feature importances. No features found or "
                "regressors do not have feature_importances_ or coef_ attributes."
            )
            self.feature_importances_ = pd.DataFrame(
                columns=["feature", "importance", "error_contribution", "selection_frequency"]
            )
            return

        combined_importance = {}

        for feature in all_features:
            error_imp = importances.get(feature, 0)
            freq_imp = selection_frequency.get(feature, 0)
            # Weighted combination: error contribution (70%) + selection frequency (30%)
            combined_importance[feature] = 0.7 * error_imp + 0.3 * freq_imp

        # Create DataFrame
        self.feature_importances_ = (
            pd.DataFrame(
                [
                    {
                        "feature": feature,
                        "importance": combined_importance[feature],
                        "error_contribution": importances.get(feature, 0),
                        "selection_frequency": selection_frequency.get(feature, 0),
                    }
                    for feature in combined_importance.keys()
                ]
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        logger.debug(f"Calculated feature importances for {len(combined_importance)} features")

    def fit_predict(
        self, df: Union[pd.DataFrame, np.ndarray], return_class: bool = False
    ) -> Union[pd.Series, np.ndarray]:
        """
        Train PSOD and return outlier predictions.

        Parameters
        ----------
        df : pd.DataFrame or np.ndarray
            Data to detect outliers from.
        return_class : bool, default=False
            Return class labels if True, else return outlier scores.

        Returns
        -------
        pd.Series or np.ndarray
            Outlier predictions. Returns ndarray if input is ndarray.
        """
        logger.info("Starting PSOD fit_predict")

        # Ensure no stale state from previous fits
        self._reset_fit_state()

        # Reset random generator for reproducibility
        self.random_generator = np.random.default_rng(self.random_seed)

        # Convert to DataFrame if necessary
        input_was_array = isinstance(df, np.ndarray)
        df = self._to_dataframe(df)
        df = self._convert_datetime_columns(df)

        # Add input validation
        self._validate_input(df)

        # Handle missing values before fitting
        df = self._handle_missing_values(df, is_training=True)
        logger.debug(f"Training data shape after preprocessing: {df.shape}")

        if isinstance(self.cat_columns, list):
            numeric_df = df.drop(self.cat_columns, axis=1).select_dtypes(include=[np.number])
            self.cols_with_var = self.remove_zero_variance(numeric_df)
            df = df.loc[:, self.cols_with_var + self.cat_columns]
        else:
            numeric_df = df.select_dtypes(include=[np.number])
            self.cols_with_var = self.remove_zero_variance(numeric_df)
            df = df.loc[:, self.cols_with_var]

        # Store feature names and count actually used for fitting
        self.feature_names_ = list(df.columns)
        self.n_features_in_ = len(df.columns)
        logger.debug(f"Using {self.n_features_in_} features for training: {self.feature_names_}")

        df_scores = df.copy()
        self.get_range_cols(df)

        if isinstance(self.cat_columns, list):
            loop_cols = df.drop(self.cat_columns, axis=1).select_dtypes(include=[np.number]).columns
            df[loop_cols] = self.fit_transform_numeric_data(df[loop_cols])
        else:
            loop_cols = df.select_dtypes(include=[np.number]).columns
            df = self.fit_transform_numeric_data(df)

        # Initialize prediction errors storage
        self.prediction_errors_ = {}

        # Add parallel processing option using joblib
        from joblib import Parallel, delayed

        def fit_single_regressor(enum_col_tuple):
            enum, col = enum_col_tuple
            logger.debug(f"Fitting regressor for column: {col}")

            # Create a deterministic random generator for this regressor
            # to ensure reproducibility in parallel execution
            local_rng = np.random.default_rng(self.random_seed + enum)

            # Use local random generator for column selection
            nb_cols_options = df.drop(columns=[col]).columns
            if len(nb_cols_options) == 0:
                warnings.warn(
                    "No columns available for selection. Cannot build regression model.",
                    UserWarning,
                )
                chosen_columns = []
            else:
                if not self._min_cols_chosen_int or not self._max_cols_chosen_int:
                    nb_cols = 0
                elif self._min_cols_chosen_int == self._max_cols_chosen_int:
                    nb_cols = self._min_cols_chosen_int
                else:
                    nb_cols = local_rng.choice(
                        np.arange(self._min_cols_chosen_int, self._max_cols_chosen_int + 1),
                        1,
                        replace=False,
                    )[0]
                nb_cols = min(nb_cols, len(nb_cols_options))
                chosen_columns = local_rng.choice(nb_cols_options, nb_cols, replace=False).tolist()

            # Handle case where no columns are available (single column dataset)
            if not chosen_columns:
                # Return default values - no error, just median scores
                default_error = pd.Series([0.5] * len(df_scores), index=df_scores.index)
                return (col, [], None, None, default_error, default_error)

            temp_df = df.copy()
            chosen_cat_cols = (
                self.col_intersection(self.cat_columns, chosen_columns)
                if isinstance(self.cat_columns, list)
                else chosen_columns
            )
            corr_cols = self.correlation_feature_selection(temp_df, col)
            corr_cols = self.col_intersection(corr_cols, chosen_columns)
            chosen_columns = corr_cols if corr_cols else chosen_columns

            idx = df_scores.sample(frac=self.sample_frac, random_state=enum, replace=True).index

            cat_encoder = None
            if isinstance(self.cat_columns, list):
                cat_encoder = self.cat_encoder(cols=chosen_cat_cols)
                (
                    cat_encoder.fit(temp_df.loc[idx, chosen_cat_cols], temp_df.loc[idx, col])
                    if self.cat_encode_on_sample
                    else cat_encoder.fit(temp_df[chosen_cat_cols], temp_df[col])
                )
                transformed_cat_cols = cat_encoder.transform(temp_df[chosen_cat_cols], temp_df[col])
                temp_df = temp_df.drop(columns=chosen_cat_cols).reset_index(drop=True)
                temp_df = pd.concat([temp_df, transformed_cat_cols.reset_index(drop=True)], axis=1)

                # Update chosen_columns to replace categorical columns with transformed columns
                chosen_columns = [c for c in chosen_columns if c not in chosen_cat_cols]
                chosen_columns.extend(transformed_cat_cols.columns.tolist())

            # Fit regressor
            if "n_jobs" in self.base_learner().get_params().keys():
                learner_params = {
                    **self.learner_kwargs,
                    "n_jobs": 1,
                }  # Use n_jobs=1 to avoid nested parallelism
                reg = self.base_learner(**learner_params).fit(
                    temp_df.loc[idx, chosen_columns], temp_df.loc[idx, col]
                )
            else:
                reg = self.base_learner(**self.learner_kwargs).fit(
                    temp_df.loc[idx, chosen_columns], temp_df.loc[idx, col]
                )

            predictions = reg.predict(temp_df[chosen_columns])
            prediction_error = abs(temp_df[col] - predictions)
            overall_error = prediction_error.mean()
            normalized_error = (
                prediction_error / overall_error if overall_error > 0 else prediction_error
            )

            return (
                col,
                chosen_columns,
                reg,
                cat_encoder,
                normalized_error,
                prediction_error,
            )

        # Process regressors (use single thread if n_jobs=1, otherwise parallel)
        if self.n_jobs == 1:
            results = []
            for enum, col in enumerate(loop_cols):
                results.append(fit_single_regressor((enum, col)))
        else:
            # Use parallel processing
            n_jobs_to_use = self.n_jobs if self.n_jobs > 0 else -1
            results = Parallel(n_jobs=n_jobs_to_use, backend="threading")(
                delayed(fit_single_regressor)((enum, col)) for enum, col in enumerate(loop_cols)
            )

        # Process results
        for (
            col,
            chosen_columns,
            reg,
            cat_encoder,
            normalized_error,
            prediction_error,
        ) in results:
            self.chosen_columns[col] = chosen_columns
            self.regressors[col] = reg
            if cat_encoder is not None:
                self.cat_encoders[col] = cat_encoder
            df_scores[col] = normalized_error
            self.prediction_errors_[col] = prediction_error.values

        df_scores = self.drop_cat_columns(df_scores)
        logger.debug(
            f"Calculated anomaly scores. Mean={df_scores['anomaly'].mean():.6f}, "
            f"Std={df_scores['anomaly'].std():.6f}"
        )

        # Store score distribution stats for thresholding
        self.pred_distribution_stats = {
            "mean_score": df_scores["anomaly"].mean(),
            "std_score": df_scores["anomaly"].std(),
        }

        # Mark as fitted before calculating feature importances and thresholds
        self._is_fitted = True

        # Adjust threshold based on contamination, if specified
        if self.contamination is not None:
            self.set_contamination_threshold(self.contamination)
            logger.debug(f"Adjusted stdevs_to_outlier to {self.stdevs_to_outlier:.6f}")

        # Calculate and store feature importances
        self._calculate_feature_importances()

        # Log fitting completion
        outlier_count = sum(self.make_outlier_classes(df_scores, use_trained_stats=True) == 1)
        logger.info(
            f"Fitting completed. Detected {outlier_count} outliers out of {len(df)} samples."
        )

        result = (
            self.make_outlier_classes(df_scores, use_trained_stats=False)
            if return_class
            else df_scores["anomaly"]
        )

        # Return numpy array if input was numpy array (sklearn compatibility)
        return result.values if input_was_array else result  # type: ignore[return-value]

    def predict(
        self,
        df: Union[pd.DataFrame, np.ndarray],
        return_class: bool = False,
        use_trained_stats: bool = True,
    ) -> Union[pd.Series, np.ndarray]:
        """
        Predict outliers using a trained PSOD instance on new data.

        Parameters
        ----------
        df : pd.DataFrame or np.ndarray
            Data to predict outliers from.
        return_class : bool, default=False
            Return class labels if True, else return outlier scores.
        use_trained_stats : bool, default=True
            Use mean and std of prediction errors from training if True.

        Returns
        -------
        pd.Series or np.ndarray
            Outlier predictions. Returns ndarray if input is ndarray.
        """
        # Convert to DataFrame if necessary
        input_was_array = isinstance(df, np.ndarray)
        df = self._to_dataframe(df)
        df = self._convert_datetime_columns(df)
        logger.info("Starting PSOD predict")

        # Check if model is fitted
        if not self._is_fitted:
            raise ValueError(
                "Model must be fitted before making predictions. Call fit_predict first."
            )

        # Add input validation (prediction mode, no min_samples check)
        self._validate_input(df, is_training=False)

        # Handle missing values using fitted imputer
        df = self._handle_missing_values(df, is_training=False)
        logger.debug(f"Prediction data shape after preprocessing: {df.shape}")

        df = df.loc[
            :,
            self.cols_with_var + (self.cat_columns if isinstance(self.cat_columns, list) else []),
        ]
        df_scores = df.copy()

        loop_cols = (
            df.drop(columns=self.cat_columns).select_dtypes(include=[np.number]).columns
            if isinstance(self.cat_columns, list)
            else df.select_dtypes(include=[np.number]).columns
        )
        df[loop_cols] = self.transform_numeric_data(df[loop_cols])

        for enum, col in enumerate(loop_cols):
            try:
                temp_df = df.copy()
                chosen_cat_cols = (
                    self.col_intersection(self.cat_columns, self.chosen_columns[col])
                    if isinstance(self.cat_columns, list)
                    else self.chosen_columns[col]
                )
                if isinstance(self.cat_columns, list) and chosen_cat_cols:
                    if col not in self.cat_encoders:
                        logger.warning(
                            f"No categorical encoder found for {col}. Skipping categorical transformation."
                        )
                    else:
                        transformed_cat_cols = self.cat_encoders[col].transform(
                            temp_df[chosen_cat_cols]
                        )
                        temp_df = temp_df.drop(columns=chosen_cat_cols).reset_index(drop=True)
                        temp_df = pd.concat(
                            [temp_df, transformed_cat_cols.reset_index(drop=True)], axis=1
                        )

                if col not in self.chosen_columns:
                    logger.warning(f"No chosen columns found for {col}. Skipping prediction.")
                    continue

                predictions = self.regressors[col].predict(temp_df[self.chosen_columns[col]])
                prediction_error = abs(temp_df[col] - predictions)
                overall_error = prediction_error.mean()
                df_scores[col] = (
                    prediction_error / overall_error if overall_error > 0 else prediction_error
                )
            except Exception as e:
                logger.error(f"Error predicting for column {col}: {str(e)}. Using zero scores.")
                df_scores[col] = 0.0
                continue

        df_scores = self.drop_cat_columns(df_scores)

        # Validate that we have valid scores
        if df_scores["anomaly"].isnull().any():
            logger.warning(
                f"Found {df_scores['anomaly'].isnull().sum()} null anomaly scores. "
                "Filling with mean score."
            )
            df_scores["anomaly"].fillna(df_scores["anomaly"].mean(), inplace=True)

        result = (
            self.make_outlier_classes(df_scores, use_trained_stats=use_trained_stats)
            if return_class
            else df_scores["anomaly"]
        )
        logger.info(f"Prediction completed for {len(df)} samples.")

        # Return numpy array if input was numpy array (sklearn compatibility)
        return result.values if input_was_array else result  # type: ignore[return-value]

    def save_model(self, filepath: str, format: str = "pickle") -> None:
        """
        Save the fitted model to disk.

        Parameters
        ----------
        filepath : str
            Path to save the model.
        format : str, default='pickle'
            Format to save ('pickle' or 'joblib').
        """
        logger.info(f"Saving PSOD model to {filepath} (format={format})")
        if not self._is_fitted:
            raise ValueError("Model must be fitted before saving.")

        # Create directory if needed
        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
            exist_ok=True,
        )

        model_data = {
            "model": self,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model_type": "PSOD",
                "version": "1.0",
                "n_features": self.n_features_in_,
                "feature_names": self.feature_names_,
            },
        }

        if format == "pickle":
            if not filepath.endswith(".pkl"):
                filepath += ".pkl"
            with open(filepath, "wb") as f:
                pickle.dump(model_data, f)
            logger.info(f"Model saved to {filepath}")
        elif format == "joblib":
            if not filepath.endswith(".joblib"):
                filepath += ".joblib"
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
        else:
            raise ValueError("Format must be 'pickle' or 'joblib'")

    @classmethod
    def load_model(cls, filepath: str, format: str = "pickle") -> "PSOD":
        """
        Load a fitted model from disk.

        .. warning::
            When using 'pickle' or 'joblib' formats, only load files from trusted sources.
            These formats can execute arbitrary code during deserialization.

        Parameters
        ----------
        filepath : str
            Path to the saved model.
        format : str, default='pickle'
            Format to load ('pickle' or 'joblib').

        Returns
        -------
        PSOD
            Loaded PSOD model.

        Raises
        ------
        FileNotFoundError
            If the model file does not exist.
        ValueError
            If the file format is invalid or the model is corrupted.
        """
        logger.info(f"Loading PSOD model from {filepath} (format={format})")
        try:
            if format == "pickle":
                if not filepath.endswith(".pkl"):
                    filepath += ".pkl"
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"Model file not found: {filepath}")
                with open(filepath, "rb") as f:
                    model_data = pickle.load(f)  # nosec B301 - Loading trusted model files only
            elif format == "joblib":
                if not filepath.endswith(".joblib"):
                    filepath += ".joblib"
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"Model file not found: {filepath}")
                model_data = joblib.load(filepath)
            else:
                raise ValueError("Format must be 'pickle' or 'joblib'")

            # Validate model data structure
            if not isinstance(model_data, dict) or "model" not in model_data:
                raise ValueError("Invalid model file format. Expected dictionary with 'model' key.")

            model = model_data["model"]
            metadata = model_data.get("metadata", {})

            # Validate that the loaded object is a PSOD instance
            if not isinstance(model, cls):
                raise ValueError(f"Loaded model is not a PSOD instance. Got {type(model)}")

            # Validate that the model is fitted
            if not model._is_fitted:
                logger.warning(
                    "Loaded model does not appear to be fitted. "
                    "You may need to call fit_predict() before using it."
                )

            logger.info(
                f"Successfully loaded PSOD model saved at {metadata.get('timestamp', 'unknown')}. "
                f"Model has {metadata.get('n_features', 'unknown')} features."
            )
            return model

        except Exception as e:
            logger.error(f"Failed to load model from {filepath}: {str(e)}")
            raise

    def get_feature_importance(self, method: str = "default") -> Optional[pd.DataFrame]:
        """
        Get feature importance scores.

        Parameters
        ----------
        method : str, default="default"
            Method for calculating feature importance.
            Options: "default", "correlation", "mutual_info"

        Returns
        -------
        pd.DataFrame or None
            DataFrame with feature importance scores, or None if model not fitted.
        """
        if not self._is_fitted:
            logger.warning("Model not fitted. Returning None for feature importance.")
            return None

        if method in ["default", "correlation"]:
            if self.feature_importances_ is None:
                self._calculate_feature_importances()
            return self.feature_importances_.copy()  # type: ignore[union-attr]
        elif method == "mutual_info":
            raise NotImplementedError("Mutual information method not yet implemented")
        else:
            raise ValueError(f"Unknown method: {method}")

    def score_samples(self, df: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Return anomaly scores for samples (sklearn compatibility).

        Parameters
        ----------
        df : pd.DataFrame or np.ndarray
            Input samples.

        Returns
        -------
        np.ndarray
            Anomaly scores.
        """
        result = self.predict(df, return_class=False)
        return result.values if isinstance(result, pd.Series) else result  # type: ignore[return-value]

    def decision_function(self, df: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Return anomaly scores for samples (sklearn compatibility).

        Parameters
        ----------
        df : pd.DataFrame or np.ndarray
            Input samples.

        Returns
        -------
        np.ndarray
            Anomaly scores.
        """
        return self.score_samples(df)

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None) -> "PSOD":
        """
        Fit the model (sklearn compatibility).

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Training data.
        y : ignored
            Not used, present for sklearn compatibility.

        Returns
        -------
        PSOD
            Fitted estimator.
        """
        self.fit_predict(X)
        return self

    def get_outlier_threshold(self) -> Union[float, Tuple[float, float]]:
        """
        Get the threshold used for outlier detection.

        Returns
        -------
        float or Tuple[float, float]
            Threshold value(s). Returns tuple for "both ends", single value otherwise.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted to get threshold.")

        if self.flag_outlier_on == "high end":
            return (
                self.pred_distribution_stats["mean_score"]
                + self.stdevs_to_outlier * self.pred_distribution_stats["std_score"]
            )
        elif self.flag_outlier_on == "low end":
            return (
                self.pred_distribution_stats["mean_score"]
                - self.stdevs_to_outlier * self.pred_distribution_stats["std_score"]
            )
        else:  # both ends
            lower = (
                self.pred_distribution_stats["mean_score"]
                - self.stdevs_to_outlier * self.pred_distribution_stats["std_score"]
            )
            upper = (
                self.pred_distribution_stats["mean_score"]
                + self.stdevs_to_outlier * self.pred_distribution_stats["std_score"]
            )
            return (lower, upper)

    def set_contamination_threshold(self, contamination: Optional[float] = None) -> None:
        """
        Set threshold based on contamination parameter.

        This method adjusts the threshold to match the expected proportion
        of outliers in the dataset.

        Parameters
        ----------
        contamination : float, optional
            Expected proportion of outliers. If None, uses self.contamination.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before setting threshold.")

        if contamination is None:
            contamination = self.contamination

        if not 0 < contamination < 1:
            raise ValueError("contamination must be between 0 and 1.")

        if self.scores is None:
            raise ValueError("No scores available. Fit the model first.")

        if self.pred_distribution_stats.get("std_score", 0) == 0:
            logger.warning(
                "Score standard deviation is zero; contamination-based threshold adjustment skipped."
            )
            return

        # Calculate threshold based on contamination
        if self.flag_outlier_on == "high end":
            threshold_value = np.percentile(self.scores, 100 * (1 - contamination))
            stdevs = (
                threshold_value - self.pred_distribution_stats["mean_score"]
            ) / self.pred_distribution_stats["std_score"]
            self.stdevs_to_outlier = stdevs  # type: ignore[assignment]
        elif self.flag_outlier_on == "low end":
            threshold_value = np.percentile(self.scores, 100 * contamination)
            stdevs = (
                self.pred_distribution_stats["mean_score"] - threshold_value
            ) / self.pred_distribution_stats["std_score"]
            self.stdevs_to_outlier = stdevs  # type: ignore[assignment]
        else:  # both ends
            # Split contamination between both ends
            lower_threshold = np.percentile(self.scores, 100 * contamination / 2)
            upper_threshold = np.percentile(self.scores, 100 * (1 - contamination / 2))
            # Use average of stdevs from both ends
            stdevs_lower = (
                self.pred_distribution_stats["mean_score"] - lower_threshold
            ) / self.pred_distribution_stats["std_score"]
            stdevs_upper = (
                upper_threshold - self.pred_distribution_stats["mean_score"]
            ) / self.pred_distribution_stats["std_score"]
            self.stdevs_to_outlier = (stdevs_lower + stdevs_upper) / 2  # type: ignore[assignment]

        logger.info(
            f"Threshold adjusted based on contamination={contamination:.3f}. "
            f"New stdevs_to_outlier={self.stdevs_to_outlier:.3f}"
        )

    def explain_outlier(self, df: pd.DataFrame, sample_idx: int, top_k: int = 5) -> Dict[str, Any]:
        """
        Explain why a sample is considered an outlier.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the sample.
        sample_idx : int
            Index of the sample to explain.
        top_k : int, default=5
            Number of top contributing features to return.

        Returns
        -------
        Dict[str, Any]
            Explanation dictionary containing:
            - sample_index: Index of the sample
            - outlier_score: Anomaly score
            - is_outlier: Boolean indicating if sample is an outlier
            - threshold: Outlier threshold(s)
            - top_contributing_features: List of (feature, contribution) tuples
            - sample_values: Dictionary of feature values for the sample
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before explaining outliers.")

        if sample_idx < 0 or sample_idx >= len(df):
            raise ValueError(
                f"sample_idx {sample_idx} is out of bounds for DataFrame of length {len(df)}"
            )

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        logger.info(f"Explaining outlier for sample at index {sample_idx}")

        # Get outlier score for this sample
        scores = self.predict(df.iloc[[sample_idx]])
        outlier_score = scores.iloc[0]  # type: ignore[union-attr]
        is_outlier = self.predict(df.iloc[[sample_idx]], return_class=True).iloc[0]  # type: ignore[union-attr]

        # Calculate feature contributions
        feature_contributions = {}
        sample_data = df.iloc[sample_idx]

        # Need to apply the same transformations as during prediction
        df_temp = df.copy()

        # Handle missing values if needed
        if df_temp.isnull().any().any():
            df_temp = self._handle_missing_values(df_temp, is_training=False)

        # Apply the same column filtering as during fitting/prediction
        df_temp = df_temp.loc[
            :,
            self.cols_with_var + (self.cat_columns if isinstance(self.cat_columns, list) else []),
        ]

        # Apply numeric transformations
        loop_cols = (
            df_temp.drop(columns=self.cat_columns).columns
            if isinstance(self.cat_columns, list)
            else df_temp.columns
        )
        df_temp[loop_cols] = self.transform_numeric_data(df_temp[loop_cols])

        for col in self.regressors.keys():
            try:
                temp_df = df_temp.copy()
                chosen_cat_cols = (
                    self.col_intersection(self.cat_columns, self.chosen_columns[col])
                    if isinstance(self.cat_columns, list)
                    else []
                )

                if chosen_cat_cols and col in self.cat_encoders:
                    transformed_cat_cols = self.cat_encoders[col].transform(
                        temp_df[chosen_cat_cols]
                    )
                    temp_df = temp_df.drop(columns=chosen_cat_cols).reset_index(drop=True)
                    temp_df = pd.concat(
                        [temp_df, transformed_cat_cols.reset_index(drop=True)],
                        axis=1,
                    )

                # Make prediction for this sample
                if col not in self.chosen_columns:
                    logger.warning(f"No chosen columns found for {col}")
                    continue

                prediction = self.regressors[col].predict(
                    temp_df.iloc[[sample_idx]][self.chosen_columns[col]]
                )
                actual_value = temp_df.iloc[sample_idx][col]
                prediction_error = abs(actual_value - prediction[0])

                feature_contributions[col] = float(prediction_error)

            except Exception as e:
                logger.warning(f"Could not calculate contribution for feature {col}: {str(e)}")
                continue

        # Sort by contribution
        if not feature_contributions:
            logger.warning(
                "Could not calculate feature contributions. Returning basic explanation."
            )
            sorted_contributions = []
        else:
            sorted_contributions = sorted(
                feature_contributions.items(), key=lambda x: x[1], reverse=True
            )

        explanation = {
            "sample_index": sample_idx,
            "outlier_score": float(outlier_score),
            "is_outlier": bool(is_outlier),
            "threshold": self.get_outlier_threshold(),
            "top_contributing_features": sorted_contributions[:top_k],
            "sample_values": sample_data.to_dict(),
            "total_features_analyzed": len(feature_contributions),
        }

        logger.info(
            f"Explanation complete. Sample is {'an outlier' if is_outlier else 'not an outlier'} "
            f"with score {outlier_score:.4f}"
        )

        return explanation
