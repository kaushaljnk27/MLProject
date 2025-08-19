import sys 
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # normalize headers: strip, lowercase, non-alnum -> underscore
        df = df.copy()
        df.columns = (
            df.columns.str.strip()
                      .str.lower()
                      .str.replace(r'[^0-9a-z]+', '_', regex=True)
        )
        return df

    def get_data_transformer_object(self):
        """
        Build the preprocessing pipeline.
        """
        try:
            # Use normalized (lowercase + underscores) names
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                "gender",
                "race_ethnicity",                # <-- corrected spelling
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Keep it simple & robust: one-hot only, ignore unknowns
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                # If you want dense output to concatenate with numeric seamlessly:
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                # If you *really* want to scale OHE output, add:
                # ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_columns),
                    ("cat", cat_pipeline, categorical_columns),
                ],
                remainder="drop",
                verbose_feature_names_out=False,
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)

            # Normalize columns to match our lists
            train_df = self._normalize_columns(train_df)
            test_df  = self._normalize_columns(test_df)

            logging.info("Read Train and Test data completed, columns normalized.")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"          # normalized target
            numerical_columns  = ["writing_score", "reading_score"]

            # Sanity checks before selection
            required_cols = set([target_column_name] + numerical_columns + [
                "gender", "race_ethnicity", "parental_level_of_education",
                "lunch", "test_preparation_course"
            ])

            missing_train = [c for c in required_cols if c not in train_df.columns]
            missing_test  = [c for c in required_cols if c not in test_df.columns]

            if missing_train:
                raise CustomException(
                    ValueError(f"Missing columns in TRAIN after normalization: {missing_train}\nAvailable: {list(train_df.columns)}"),
                    sys
                )
            if missing_test:
                raise CustomException(
                    ValueError(f"Missing columns in TEST after normalization: {missing_test}\nAvailable: {list(test_df.columns)}"),
                    sys
                )

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr  = preprocessing_obj.transform(input_feature_test_df)

            # Ensure arrays (OneHotEncoder with sparse_output=False already gives dense)
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr  = np.c_[input_feature_test_arr,  np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            # bubble up with context
            raise CustomException(e, sys)
