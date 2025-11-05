import sys, os
import pandas as pd
import numpy as np
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from networksecurity.constant.training_pipeline import TARGET_COLUMN
from networksecurity.constant.training_pipeline import DATA_TRANSFORMATION_INPUT_PARAMS
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from networksecurity.utils.main_utils.utils import save_numpy_arry_data, save_object


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        """
        Purpose:
          - Wire in the validated dataset locations (train/test CSV paths)
          - Wire in the output locations for transformed arrays and the fitted transformer

        Args:
          data_validation_artifact: carries valid_train_file_path and valid_test_file_path
          data_transformation_config: carries transformed_* file paths for outputs
        """
        try:
            # Store references for later steps in the pipeline
            self.data_validation_artifact: DataValidationArtifact = data_validation_artifact
            self.data_transformation_config: DataTransformationConfig = data_transformation_config

        except Exception as e:
            # Wrap any unexpected initialization errors with project-specific exception
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Read a CSV file into a pandas DataFrame.

        Raises:
          NetworkSecurityException on any IO/parse error
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Build and return the preprocessing pipeline.

        Current pipeline:
          - KNNImputer with parameters from DATA_TRANSFORMATION_INPUT_PARAMS

        Returns:
          A scikit-learn Pipeline ready to fit/transform input features.
        """
        logging.info("Entered get_data_transformer_object")
        try:
            # Initialize imputers/transformers per config (e.g., n_neighbors, weights, etc.)
            imputer: KNNImputer = KNNImputer(**DATA_TRANSFORMATION_INPUT_PARAMS)
            logging.info(
                f"Initialized KNNImputer with parameters: {DATA_TRANSFORMATION_INPUT_PARAMS}"
            )

            # Wrap in a Pipeline to make it easy to extend later (scaling, encoding, etc.)
            processor: Pipeline = Pipeline([("imputer", imputer)])
            return processor

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Orchestrate data transformation:
          1) Load validated train/test CSVs
          2) Split features/target
          3) Fit the preprocessing pipeline on train features
          4) Transform train/test features
          5) Concatenate transformed features with target to produce numpy arrays
          6) Persist arrays and the fitted transformer object
          7) Return a DataTransformationArtifact pointing to the saved outputs
        """
        logging.info("Entered the data transformation pipeline")
        try:
            # --- 1) Load validated datasets ---
            logging.info("Loading validated train/test data")
            train_df = DataTransformation.read_data(
                file_path=self.data_validation_artifact.valid_train_file_path
            )
            test_df = DataTransformation.read_data(
                file_path=self.data_validation_artifact.valid_test_file_path
            )

            # --- 2) Split input features vs. target ---
            # Drop the target from the feature matrices; keep a separate target Series
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            # Log basic shapes for sanity checks
            logging.info(
                f"Train shapes -> X: {input_feature_train_df.shape}, y: {target_feature_train_df.shape}"
            )
            logging.info(
                f"Test  shapes -> X: {input_feature_test_df.shape}, y: {target_feature_test_df.shape}"
            )

            # --- 3) Fit the transformer on TRAIN features only ---
            preprocessor = self.get_data_transformer_object()
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            logging.info("Fitted preprocessing pipeline on training features")

            # --- 4) Transform both TRAIN and TEST features with the fitted transformer ---
            transformed_input_train_feature = preprocessor_object.transform(
                input_feature_train_df
            )
            transformed_input_test_feature = preprocessor_object.transform(
                input_feature_test_df
            )

            # --- 5) Concatenate transformed features with the target into final arrays ---
            # np.c_ horizontally stacks arrays/columns: [X | y]
            train_arr = np.c_[
                transformed_input_train_feature, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                transformed_input_test_feature, np.array(target_feature_test_df)
            ]
            logging.info(
                f"Transformed arrays -> train: {train_arr.shape}, test: {test_arr.shape}"
            )

            # --- 6) Persist transformed arrays and the fitted transformer object ---
            save_numpy_arry_data(
                file_path=self.data_transformation_config.transformed_train_file_path,
                array=train_arr,
            )
            save_numpy_arry_data(
                file_path=self.data_transformation_config.transformed_test_file_path,
                array=test_arr,
            )
            save_object(
                file_path=self.data_transformation_config.transformed_object_file_path,
                obj=preprocessor_object,
            )
            logging.info(
                "Saved transformed train/test arrays and preprocessor object to disk"
            )

            # --- 7) Build and return the artifact with output paths ---
            data_transformation_artifact: DataTransformationArtifact = (
                DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                )
            )

            return data_transformation_artifact

        except Exception as e:
            # Wrap and re-raise for consistent error reporting (file name & line number)
            raise NetworkSecurityException(e, sys)
