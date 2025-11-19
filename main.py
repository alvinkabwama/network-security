import sys  # Provides access to interpreter-specific variables and functions (used here for exception handling)
from networksecurity.logging import logger  # Project-level logger for logging pipeline steps
from networksecurity.exception.exception import NetworkSecurityException  # Custom exception class for consistent error handling

# Pipeline components
from networksecurity.components.data_ingestion import DataIngestion  # Handles reading raw data and splitting into train/test
from networksecurity.components.data_validation import DataValidation  # Validates data quality and schema
from networksecurity.components.data_transformation import DataTransformation  # Transforms data (scaling, encoding, etc.)
from networksecurity.components.model_trainer import ModelTrainer  # Trains and evaluates the ML model

# Configuration entities for each pipeline stage
from networksecurity.entity.config_entity import (
    DataIngestionConfig, 
    TrainingPipelineConfig, 
    DataValidaitonConfig, 
    DataTransformationConfig, 
    ModelTrainerConfig)


# Script entry point: runs each pipeline component step-by-step
if __name__ == "__main__":
    try:
        # Instantiate main training pipeline config with timestamped artifact directories
        training_pipeline_config = TrainingPipelineConfig()

        # Create configuration objects for each pipeline stage
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_validation_config = DataValidaitonConfig(training_pipeline_config=training_pipeline_config)
        data_transformation_config = DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        model_trainer_config = ModelTrainerConfig(training_pipeline_config=training_pipeline_config)

        # -------------------- DATA INGESTION --------------------
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        logger.logging.info("Initiate the data Ingestion")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)  # Print artifact with train/test paths for debug

        # -------------------- DATA VALIDATION --------------------
        data_validation = DataValidation(
            data_validation_config=data_validation_config,
            data_ingestion_artifact=data_ingestion_artifact
        )
        logger.logging.info("Initiate the data Validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        print(data_validation_artifact)  # Print validation results and drift report path

        # -------------------- DATA TRANSFORMATION --------------------
        data_transformation = DataTransformation(
            data_transformation_config=data_transformation_config,
            data_validation_artifact=data_validation_artifact
        )
        logger.logging.info("Initiate the data Tranformation")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)  # Print transformed data and preprocessing object paths

        # -------------------- MODEL TRAINING --------------------
        model_trainer = ModelTrainer(
            model_trainer_config=model_trainer_config,
            data_transformation_artifact=data_transformation_artifact
        )
        logger.logging.info("Initiating the Model Training Step")
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        print(model_trainer_artifact)  # Print model metrics and trained model path
        
    except Exception as e:
        # Wrap any unhandled exception into NetworkSecurityException for better traceability
        raise NetworkSecurityException(e, sys)
