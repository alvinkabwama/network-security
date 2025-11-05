import sys
from networksecurity.logging import logger
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, DataValidaitonConfig


if __name__ == "__main__":
    try:

        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_validation_config =  DataValidaitonConfig(training_pipeline_config=training_pipeline_config)


        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        logger.logging.info("Initiate the data Ingestion")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)

        data_validation = DataValidation(data_validation_config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)
        data_validation_artifact = data_validation.initiate_data_validation()
        print(data_validation_artifact)

        
       
    except Exception as e:
        raise NetworkSecurityException(e, sys)