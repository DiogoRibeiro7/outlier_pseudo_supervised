from typing import List, Union, Dict, Type
import warnings
import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from tqdm import tqdm
import gc
from category_encoders import BaseNEncoder, TargetEncoder


class PSOD:
    """
    Get outlier predictions using a pseudo-supervised approach.

    :param n_jobs: Number of cores used for the regressor. Default is -1 (all cores).
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
    :param base_learner: A regressor class to be used as the base learner for predictions.
    :param cat_encoder: A categorical encoder class from category_encoders to be used for encoding categorical features.
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
        cat_encoder: Type[BaseNEncoder] = TargetEncoder
    ):
        self.cat_columns = cat_columns
        self.cat_encoders: Dict[str, BaseNEncoder] = {}
        self.numeric_encoders: Union[PowerTransformer, None] = None
        self.regressors: Dict[str, RegressorMixin] = {}
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
        self.base_learner = base_learner
        self.cat_encoder = cat_encoder

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
        - base_learner: {self.base_learner.__name__}
        - cat_encoder: {self.cat_encoder.__name__}
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
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        return [
            col for col in numerical_cols
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
        elif self.transform_algorithm == "yeo-johnson":
            scaler = PowerTransformer(method="yeo-johnson")
            df_transformed = scaler.fit_transform(df)
            self.numeric_encoders = scaler
            return pd.DataFrame(df_transformed, columns=df.columns)
        return df

    def transform_numeric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform numeric data based on the fitted transformation algorithm.

        :param df: DataFrame to transform.
        :return: Transformed DataFrame.
        """
        if self.transform_algorithm == "logarithmic":
            return np.log1p(df)
        elif self.transform_algorithm == "yeo-johnson" and self.numeric_encoders:
            df_transformed = self.numeric_encoders.transform(df)
            return pd.DataFrame(df_transformed, columns=df.columns)
        return df

    def remove_zero_variance(self, df: pd.DataFrame) -> List[str]:
        """
        Remove columns with zero variance.

        :param df: DataFrame to check for zero variance.
        :return: List of columns with non-zero variance.
        """
        var_data = df.var()
        return var_data[var_data != 0].index.to_list()

    def fit_predict(self, df: pd.DataFrame, return_class: bool = False) -> pd.Series:
        """
        Train PSOD and return outlier predictions.

        :param df: DataFrame to detect outliers from.
        :param return_class: Return class labels if True, else return outlier scores.
        :return: Outlier predictions as a Pandas Series.
        """
        if isinstance(self.cat_columns, list):
            self.cols_with_var = self.remove_zero_variance(df.drop(self.cat_columns, axis=1))
            df = df.loc[:, self.cols_with_var + self.cat_columns]
        else:
            self.cols_with_var = self.remove_zero_variance(df)
            df = df.loc[:, self.cols_with_var]

        df_scores = df.copy()
        self.get_range_cols(df)
        if isinstance(self.cat_columns, list):
            loop_cols = df.drop(self.cat_columns, axis=1).columns
            df[loop_cols] = self.fit_transform_numeric_data(df[loop_cols])
        else:
            loop_cols = df.columns
            df = self.fit_transform_numeric_data(df)

        for enum, col in tqdm(enumerate(loop_cols), total=len(loop_cols)):
            self.chosen_columns[col] = self.chose_random_columns(df.drop(columns=[col]))
            temp_df = df.copy()
            chosen_cat_cols = self.col_intersection(self.cat_columns, self.chosen_columns[col]) if isinstance(self.cat_columns, list) else self.chosen_columns[col]
            corr_cols = self.correlation_feature_selection(temp_df, col)
            corr_cols = self.col_intersection(corr_cols, self.chosen_columns[col])
            self.chosen_columns[col] = corr_cols if corr_cols else self.chosen_columns[col]
            idx = df_scores.sample(frac=self.sample_frac, random_state=enum, replace=True).index

            if isinstance(self.cat_columns, list):
                enc = self.cat_encoder(cols=chosen_cat_cols)
                enc.fit(temp_df.loc[idx, chosen_cat_cols], temp_df.loc[idx, col]) if self.cat_encode_on_sample else enc.fit(temp_df[chosen_cat_cols], temp_df[col])
                transformed_cat_cols = enc.transform(temp_df[chosen_cat_cols], temp_df[col])
                temp_df = temp_df.drop(columns=chosen_cat_cols).reset_index(drop=True)
                temp_df = pd.concat([temp_df, transformed_cat_cols.reset_index(drop=True)], axis=1)
            
            if 'n_jobs' in self.base_learner().get_params().keys():
                reg = self.base_learner(n_jobs=self.n_jobs).fit(temp_df.loc[idx, self.chosen_columns[col]], temp_df.loc[idx, col])
            else:
                reg = self.base_learner().fit(temp_df.loc[idx, self.chosen_columns[col]], temp_df.loc[idx, col])
            predictions = reg.predict(temp_df[self.chosen_columns[col]])
            prediction_error = abs(temp_df[col] - predictions)
            overall_error = prediction_error.mean()
            df_scores[col] = prediction_error / overall_error  # Apply regularization by weighting prediction error

            self.regressors[col] = reg
            if isinstance(self.cat_columns, list):
                self.cat_encoders[col] = enc
                del enc

            del temp_df, idx, reg
            gc.collect()

        df_scores = self.drop_cat_columns(df_scores)
        return self.make_outlier_classes(df_scores, use_trained_stats=False) if return_class else df_scores["anomaly"]

    def predict(self, df: pd.DataFrame, return_class: bool = False, use_trained_stats: bool = True) -> pd.Series:
        """
        Predict outliers using a trained PSOD instance on new data.

        :param df: DataFrame to predict outliers from.
        :param return_class: Return class labels if True, else return outlier scores.
        :param use_trained_stats: Use mean and std of prediction errors from training if True.
        :return: Outlier predictions as a Pandas Series.
        """
        df = df.loc[:, self.cols_with_var + (self.cat_columns if isinstance(self.cat_columns, list) else [])]
        df_scores = df.copy()

        loop_cols = df.drop(columns=self.cat_columns).columns if isinstance(self.cat_columns, list) else df.columns
        df[loop_cols] = self.transform_numeric_data(df[loop_cols])

        for enum, col in tqdm(enumerate(loop_cols)):
            temp_df = df.copy()
            chosen_cat_cols = self.col_intersection(self.cat_columns, self.chosen_columns[col]) if isinstance(self.cat_columns, list) else self.chosen_columns[col]
            if isinstance(self.cat_columns, list):
                transformed_cat_cols = self.cat_encoders[col].transform(temp_df[chosen_cat_cols])
                temp_df = temp_df.drop(columns=chosen_cat_cols).reset_index(drop=True)
                temp_df = pd.concat([temp_df, transformed_cat_cols.reset_index(drop=True)], axis=1)
            predictions = self.regressors[col].predict(temp_df[self.chosen_columns[col]])
            prediction_error = abs(temp_df[col] - predictions)
            overall_error = prediction_error.mean()
            df_scores[col] = prediction_error / overall_error  # Apply regularization by weighting prediction error

        df_scores = self.drop_cat_columns(df_scores)
        return self.make_outlier_classes(df_scores, use_trained_stats=use_trained_stats) if return_class else df_scores["anomaly"]


import pandas as pd
from sklearn.linear_model import Ridge
from category_encoders import OneHotEncoder

# Sample data creation
data = {
    'A': [1, 2, 3, 4, 5, 100],
    'B': [10, 20, 30, 40, 50, 1000],
    'C': ['cat', 'dog', 'cat', 'dog', 'cat', 'dog']
}
df = pd.DataFrame(data)

# Initialize PSOD with Ridge regression as the base learner and OneHotEncoder for categorical encoding
psod = PSOD(
    n_jobs=1,
    cat_columns=['C'],
    min_cols_chosen=0.5,
    max_cols_chosen=1.0,
    stdevs_to_outlier=1.96,
    sample_frac=1.0,
    correlation_threshold=0.05,
    transform_algorithm='logarithmic',
    random_seed=42,
    cat_encode_on_sample=False,
    flag_outlier_on='both ends',
    base_learner=Ridge,
    cat_encoder=OneHotEncoder
)

# Fit and predict outliers
outlier_scores = psod.fit_predict(df)
print(outlier_scores)

# Predict outliers on new data
new_data = {
    'A': [2, 3, 4, 50],
    'B': [15, 25, 35, 500],
    'C': ['cat', 'dog', 'cat', 'dog']
}
new_df = pd.DataFrame(new_data)

outlier_scores_new = psod.predict(new_df)
print(outlier_scores_new)
