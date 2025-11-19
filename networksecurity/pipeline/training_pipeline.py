from networksecurity.logging.logger import logging  # Project-level configured logger
import os, sys  # OS utilities and sys for exception detail
from networksecurity.exception.exception import NetworkSecurityException  # Custom exception class
from networksecurity.components.data_ingestion import DataIngestion  # Data ingestion component
from networksecurity.components.data_validation import DataValidation  # Data validation component
from networksecurity.components.data_transformation import DataTransformation  # Data transformation component
from networksecurity.components.model_trainer import ModelTrainer  # Model training component
from networksecurity.constant.training_pipeline import TRAINING_BUCKET_NAME  # S3 training bucket name
from networksecurity.cloud.s3_syncer import S3Sync  # Helper to sync folders with S3

from networksecurity.entity.config_entity import (
    DataIngestionConfig, 
    TrainingPipelineConfig, 
    DataValidaitonConfig, 
    DataTransformationConfig, 
    ModelTrainerConfig)

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact,
    ModelTrainerArtifact
)

# Orchestrates the entire ML training pipeline end-to-end
class TrainingPipeline:
    def __init__(self):
        # Initialize the root training pipeline config (timestamped artifact dirs, etc.)
        self.training_pipeline_config = TrainingPipelineConfig()
        # Initialize S3 sync helper for uploading artifacts and models
        self.s3_sync = S3Sync()

    # Step 1: Start data ingestion process
    def start_data_ingestion(self):
        try:
            # Build configuration for data ingestion using the global pipeline configuration
            data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start Data Ingestion")
            # Create DataIngestion component with its config
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            # Trigger the actual ingestion logic
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion completed and artifact is {data_ingestion_artifact}")
            # Return artifact containing file paths produced by ingestion
            return data_ingestion_artifact

        except Exception as e:
            # Wrap and re-raise any exception as NetworkSecurityException with traceback info
            raise NetworkSecurityException(e, sys)
        
    # Step 2: Start data validation process
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact):
        try:
            # Build configuration for data validation
            data_validation_config = DataValidaitonConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start Data Validation")
            # Initialize DataValidation component with ingestion outputs and validation config
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=data_validation_config
            )
            # Run validation (schema checks, drift checks, etc.)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"Data Validation completed and artifact is {data_validation_artifact}")
            # Return artifact containing paths to valid/invalid data and drift report
            return data_validation_artifact
        except Exception as e:
            # Convert any failure into NetworkSecurityException
            raise NetworkSecurityException(e,sys)
        
    # Step 3: Start data transformation process
    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact):
        try:
            # Build configuration for data transformation
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start Data Transformation")
            # Initialize DataTransformation component with validated data and config
            data_transformation = DataTransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=data_transformation_config
            )
            # Execute transformation (scaling, encoding, feature engineering, etc.)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info(f"Data Transformation completed and artifact is {data_transformation_artifact}")
            # Return artifact containing transformed arrays and preprocessing object path
            return data_transformation_artifact
        except Exception as e:
            # Wrap exceptions from transformation
            raise NetworkSecurityException(e, sys)
    

    # Step 4: Start model training process
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact):
        try:
            # Build configuration for model trainer
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start Model Trainer")
            # Initialize ModelTrainer with config and transformed data artifacts
            model_trainer = ModelTrainer(
                model_trainer_config=model_trainer_config, 
                data_transformation_artifact=data_transformation_artifact
            )
            # Run model training/tuning and evaluation
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info(f"Data Transformation completed and artifact is {model_trainer_artifact}")
            # Return artifact containing trained model path and metrics
            return model_trainer_artifact
        except Exception as e:
            # Wrap model training exceptions
            raise NetworkSecurityException(e, sys)
        
     ## local artifact is going to s3 bucket    
    def sync_artifact_dir_to_s3(self):
        try:
            # Build S3 path for storing pipeline artifacts for this timestamped run
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            # Upload the entire local artifact directory to S3
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.artifact_dir, aws_bucket_url=aws_bucket_url)
        except Exception as e:
            # Wrap sync errors into NetworkSecurityException
            raise NetworkSecurityException(e,sys)
        
    ## local final model is going to s3 bucket 
    def sync_saved_model_dir_to_s3(self):
        try:
            # Build S3 path for storing final trained model(s) for this run
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/final_model/{self.training_pipeline_config.timestamp}"
            # Upload the local model directory to S3
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.model_dir, aws_bucket_url=aws_bucket_url)
        except Exception as e:
            # Wrap sync errors into NetworkSecurityException
            raise NetworkSecurityException(e,sys)
                       
    # Orchestrates the full pipeline: ingestion -> validation -> transformation -> training -> sync to S3
    def run_pipeline(self):
        try:
            # 1. Ingest the raw data and produce ingestion artifacts
            data_ingestion_artifact = self.start_data_ingestion()
            # 2. Validate the ingested data and produce validation artifacts
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            # 3. Transform the validated data and produce transformation artifacts
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            # 4. Train and evaluate the model using transformed data
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)

            # 5. Sync all artifacts and final models to S3 for storage/backups
            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_dir_to_s3()

            # Return final model training artifact (metrics + model path)
            return model_trainer_artifact
        except Exception as e:
            # Any failure at pipeline level is wrapped in NetworkSecurityException
            raise NetworkSecurityException(e, sys)
