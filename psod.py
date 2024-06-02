from typing import List, Union, Dict
import warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from tqdm import tqdm
from category_encoders import TargetEncoder
import gc


class PSOD:
    """
    Get outlier predictions using a pseudo-supervised approach.

    :param n_jobs: Number of cores used for LinearRegression. Default is -1 (all cores).
    :param cat_columns: List specifying column names of categorical features. None if no categorical features.
    :param min_cols_chosen: Minimum percentage of columns to be used for each regressor.
    :param max_cols_chosen: Maximum percentage of columns to be used for each regressor.
    :param stdevs_to_outlier: Number of standard deviations from the mean prediction error to flag as an outlier.
    :param sample_frac: Fraction of rows each bagging sample shall use.
    :param correlation_threshold: Filter out all columns with a correlation closer to 0 than the threshold.
    :param transform_algorithm: Algorithm for transforming numerical columns. One of ["logarithmic", "yeo-johnson", "none"].
    :param random_seed: Initial random seed. Each additional iteration will use a different seed.
    :param cat_encode_on_sample: Apply categorical encoding to bagging sample if True; otherwise, use the full dataset.
    :param flag_outlier_on: Specify whether to flag errors on the "low end", "both ends", or "high end" of the mean error distribution.
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
        flag_outlier_on: str = "both ends"
    ):
        self.cat_columns = cat_columns
        self.cat_encoders: Dict[str, TargetEncoder] = {}
        self.numeric_encoders: Union[PowerTransformer, None] = None
        self.regressors: Dict[str, LinearRegression] = {}
        self.n_jobs = n_jobs
        self.scores: Union[pd.Series, None] = None
        self.outlier_classes: Union[pd.Series, None] = None
        self.min_cols_chosen = min_cols_chosen
        self.max_cols_chosen = max_cols_chosen
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

        # Validate parameters
        self._validate_params()

    def _validate_params(self):
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
        if self.sample_frac > 1.0:
            warnings.warn(
                "sample_frac is set higher than 1.0. This might lead to overfitting. Recommended value is 1.",
                UserWarning
            )
        if self.min_cols_chosen < 0.3:
            warnings.warn(
                "min_cols_chosen is set to a very low value of less than 0.3. This may reduce performance.",
                UserWarning
            )
        if self.correlation_threshold > 0.15:
            warnings.warn(
                "correlation_threshold is set higher than 0.15. This may harm performance. Recommended value is 0.05.",
                UserWarning
            )

    def __str__(self):
        return f"""
        Most important parameters:
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
        """

    def get_range_cols(self, df: pd.DataFrame):
        len_cols = len(df.columns) - 1  # Exclude target column
        self.min_cols_chosen = max(int(len_cols * self.min_cols_chosen), 1)
        self.max_cols_chosen = min(int(len_cols * self.max_cols_chosen), len_cols)

    def chose_random_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Select random columns.
        
        :param df: DataFrame to select columns from.
        :return: List of chosen column names.
        """
        nb_cols = (1 if self.min_cols_chosen == self.max_cols_chosen == 1
                   else self.random_generator.choice(
                        np.arange(self.min_cols_chosen, self.max_cols_chosen) + 1, 1, replace=False
                   )[0])
        return self.random_generator.choice(df.columns, nb_cols, replace=False).tolist()

    def correlation_feature_selection(self, df: pd.DataFrame, target_col: str) -> List[str]:
        """
        Select columns based on correlation with target column.

        :param df: DataFrame to select columns from.
        :param target_col: Target column for correlation computation.
        :return: List of columns with correlation above threshold.
        """
        return [
            col for col in df.columns 
            if col != target_col and abs(df[col].corr(df[target_col])) > self.correlation_threshold
        ]

    def col_intersection(self, lst1: List[str], lst2: List[str]) -> List[str]:
        """
        Find intersection of two lists.

        :param lst1: First list.
        :param lst2: Second list.
        :return: List of intersecting elements.
        """
        return np.intersect1d(lst1, lst2).tolist()

    def make_outlier_classes(self, df_scores: pd.DataFrame, use_trained_stats=True) -> pd.Series:
        if not use_trained_stats:
            mean_score = df_scores["anomaly"].mean()
            std_score = df_scores["anomaly"].std()
            self.pred_distribution_stats = {"mean_score": mean_score, "std_score": std_score}

        conditions = []
        if self.flag_outlier_on in ["both ends", "low end"]:
            conditions.append(df_scores["anomaly"] < self.pred_distribution_stats["mean_score"] - self.stdevs_to_outlier * self.pred_distribution_stats["std_score"])
        if self.flag_outlier_on in ["both ends", "high end"]:
            conditions.append(df_scores["anomaly"] > self.pred_distribution_stats["mean_score"] + self.stdevs_to_outlier * self.pred_distribution_stats["std_score"])

        df_scores["anomaly_class"] = np.select(conditions, [1]*len(conditions), default=0)
        self.outlier_classes = df_scores["anomaly_class"]
        return df_scores["anomaly_class"]

    def drop_cat_columns(self, df_scores: pd.DataFrame) -> pd.DataFrame:
        """
        Drop categorical columns from the DataFrame and calculate anomaly score.

        :param df_scores: DataFrame with predictions.
        :return: DataFrame with anomaly score.
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

        :param df: DataFrame to transform.
        :return: Transformed DataFrame.
        """
        if self.transform_algorithm == "logarithmic":
            return np.log1p(df)
        elif self.transform_algorithm == "
