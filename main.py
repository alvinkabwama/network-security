import sys
from networksecurity.logging import logger
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.entity.config_entity import (
    DataIngestionConfig, 
    TrainingPipelineConfig, 
    DataValidaitonConfig, 
    DataTransformationConfig, 
    ModelTrainerConfig)


if __name__ == "__main__":
    try:

        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_validation_config =  DataValidaitonConfig(training_pipeline_config=training_pipeline_config)
        data_transformation_config = DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        model_trainer_config = ModelTrainerConfig(training_pipeline_config=training_pipeline_config)


        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        logger.logging.info("Initiate the data Ingestion")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)

        data_validation = DataValidation(data_validation_config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)
        logger.logging.info("Initiate the data Validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        print(data_validation_artifact)

        data_transformation = DataTransformation(data_transformation_config=data_transformation_config,data_validation_artifact=data_validation_artifact)
        logger.logging.info("Initiate the data Tranformation")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)

        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
        logger.logging.info("Initiating the Model Training Step")
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        print(model_trainer_artifact)
        
  
    except Exception as e:
        raise NetworkSecurityException(e, sys)