# Purpose: Pull a collection from MongoDB into a DataFrame, persist a feature-store CSV,
# and split into train/test CSVs. Raises NetworkSecurityException on failure.

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

# Configuration / artifact dataclasses (paths, names, etc.)
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact

import os
import sys
import pymongo
from typing import List
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env (e.g., MONGO_DB_URL)

MONGO_DB_URL = os.getenv("MONGO_DB_URL")  # MongoDB connection string

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Store config used across the ingestion steps.
        """
        try:
            self.data_ingestion_config = data_ingestion_config
            # No heavy work in __init__; only assign config.
        except Exception as e:
            # Wrap any unexpected init errors in project-specific exception.
            raise NetworkSecurityException(e, sys)
        
    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """
        Read a MongoDB collection and return it as a pandas DataFrame.
        - Drops the MongoDB '_id' column if present.
        - Replaces string 'na' with np.nan.
        """
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            # Create a client using the connection URL from the environment.
            # NOTE: for Atlas (mongodb+srv), certifi.tlsCAFile may be required.
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)

            # Select database and collection.
            collection = self.mongo_client[database_name][collection_name]
            
            # Pull all docs; convert cursor -> list -> DataFrame.
            df = pd.DataFrame(list(collection.find()))

            # Drop Mongo's internal ObjectId column if present.
            if "_id" in df.columns.to_list():
                # IMPORTANT: df.drop returns a new DataFrame; assign it back or use inplace=True.
                df = df.drop(columns=["_id"], axis=1)

            # Normalize string placeholders for missing values.
            df.replace({"na": np.nan}, inplace=True)

            return df

        except Exception as e:
            # Include file/line info using NetworkSecurityException for easier debugging.
            raise NetworkSecurityException(e, sys)            
        
    def export_data_to_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Persist the raw DataFrame into the 'feature store' CSV path.
        Returns the same DataFrame to allow chaining.
        """
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path

            # Ensure the directory exists before writing the CSV.
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            # Write the full dataset to the feature store.
            dataframe.to_csv(feature_store_file_path, index=False, header=True)

            logging.info(f"Feature store written to: {feature_store_file_path}")
            return dataframe

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test_split(self, dataframe: pd.DataFrame) -> None:
        """
        Split the DataFrame into train/test and write them to configured CSV paths.
        Files:
          - training_file_path
          - test_file_path
        """
        try:
            # Perform the split; consider adding random_state for reproducibility if desired.
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio
                # , random_state=42
            )
            logging.info("Performed train_test_split on the dataframe")

            # Ensure the parent directory for train/test paths exists.
            # Use exist_ok=True in case it already exists.
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info("Exporting the training and the test file path")

            # Write out the splits.
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_file_path, index=False, header=True)

            logging.info(
                f"Train -> {self.data_ingestion_config.training_file_path} | "
                f"Test -> {self.data_ingestion_config.test_file_path}"
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Orchestrate the ingestion steps:
          1) Load from MongoDB
          2) Persist to feature store
          3) Split and persist train/test
        Returns a DataIngestionArtifact with paths to train/test files.
        """
        try:
            # 1) Read from source (MongoDB -> DataFrame)
            dataframe = self.export_collection_as_dataframe()

            # 2) Save to feature store
            dataframe = self.export_data_to_feature_store(dataframe)

            # 3) Split and persist
            self.split_data_as_train_test_split(dataframe)

            # Build the artifact object to pass along the pipeline.
            data_ingestion_artifact = DataIngestionArtifact(
                training_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )
            return data_ingestion_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
