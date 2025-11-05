from networksecurity.entity.config_entity import DataValidaitonConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
from networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file
import pandas as pd
import os, sys

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidaitonConfig):
        """
        Wires up the artifacts/configs needed for validation:
        - data_ingestion_artifact: carries paths to train/test CSVs produced by ingestion
        - data_validation_config: carries output paths (valid/invalid dirs, drift report path, etc.)
        - schema_config: YAML schema loaded from SCHEMA_FILE_PATH
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """
        Read a CSV file into a dataframe.
        Wrapped with project exception to include file/line in errors.
        """
        try:
            dataframe = pd.read_csv(file_path)
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def validate_no_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Check that the dataframe has the expected number of columns.
        NOTE: This uses len(self.schema_config), which assumes your schema dict's
        top-level length equals the number of columns. If your schema is nested
        (e.g., {'columns': [...], 'numerical_columns': [...]}) you may want to
        change this to len(self.schema_config['columns']).
        """
        try:
            number_of_columns = len(self.schema_config)
            logging.info(f"Required number of columns {number_of_columns}")
            logging.info(f"Data frame has {len(dataframe.columns)}")

            if len(dataframe.columns) == number_of_columns:
                return True
            return False 
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> bool:
        """
        Perform a univariate KS test column-by-column between base (train) and current (test).
        - Returns True if NO drift found (all p-values >= threshold).
        - Returns False if ANY drift found (any p-value < threshold).
        Also writes a YAML report with p-values and per-column drift status.
        """
        try:
            status = True  # stays True only if no column shows drift
            report = {}

            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                test = ks_2samp(d1, d2)

                # drift if p-value < threshold
                drift_found = test.pvalue < threshold
                if drift_found:
                    status = False

                report[column] = {
                    "p_value": float(test.pvalue),
                    "drift_status": drift_found
                }

            # Write drift report to configured path, ensuring directory exists
            drift_report_file_path = self.data_validation_config.drift_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)

            write_yaml_file(file_path=drift_report_file_path, content=report)

            # 🔁 Minimal addition: return status so caller can combine with other checks
            return status

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Orchestrates validation:
          1) Read train/test CSVs
          2) Validate column counts on both
          3) Run drift detection
          4) Write 'valid' copies and emit a DataValidationArtifact with overall status
        """
        try:
            train_file_path = self.data_ingestion_artifact.training_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # 1) Load data
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            # 2) Column-count checks (per your existing method)
            train_cols_ok = self.validate_no_of_columns(dataframe=train_dataframe)
            if not train_cols_ok:
                logging.info("Train data frame does not contain all the columns")

            test_cols_ok = self.validate_no_of_columns(dataframe=test_dataframe)
            if not test_cols_ok:
                logging.info("Test data frame does not contain all the columns")

            # 3) Drift detection
            drift_ok = self.detect_dataset_drift(base_df=train_dataframe, current_df=test_dataframe)

            # ✅ Minimal change: aggregate all checks into a single overall status
            #    Overall is True only if ALL individual checks passed.
            status = bool(train_cols_ok and test_cols_ok and drift_ok)

            # 4) Persist 'valid' versions (your current behavior writes them regardless of status)
            dir_path = os.path.dirname(self.data_validation_config.valid_training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_dataframe.to_csv(self.data_validation_config.valid_training_file_path, index=False, header=True)
            test_dataframe.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            # Build and return the artifact
            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.training_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_file_path
            )

            return data_validation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
