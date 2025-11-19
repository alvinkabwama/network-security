from datetime import datetime  # Used to generate timestamped artifact directories
import os  # Used to build file/directory paths
from networksecurity.constant import training_pipeline  # Contains global pipeline constants (dir names, filenames, ratios, etc.)

# Debug prints showing configured artifact directory and pipeline name
print(training_pipeline.ARTIFACT_DIR)
print(training_pipeline.PIPELINE_NAME)


# Configuration class for the overall training pipeline
class TrainingPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        # Convert timestamp to formatted string (e.g., 03_14_2025_10_32_45)
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")

        # Name of the pipeline (from constants)
        self.pipeline_name = training_pipeline.PIPELINE_NAME

        # Root artifact directory name
        self.artifact_name = training_pipeline.ARTIFACT_DIR

        # Main artifact directory for the current run (unique timestamp)
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)

        # Directory for storing trained models (also under artifact folder)
        self.model_dir = os.path.join(self.artifact_name, timestamp)

        # Saves timestamp for referencing the run later
        self.timestamp: str = timestamp


# Configuration for Data Ingestion component
class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        # Base directory for ingestion artifacts
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME
        )

        # Path where the ingested raw feature store file will be stored
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            training_pipeline.FILE_NAME
        )

        # Final training dataset path
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TRAIN_FILE_NAME
        )

        # Final test dataset path
        self.test_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TEST_FILE_NAME
        )

        # Ratio used to split ingested data (e.g., 0.2 test)
        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO

        # MongoDB or other DB collection name to read raw data from (if used)
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME

        # Database name for raw ingestion source
        self.database_name: str = training_pipeline.DATA_INGESTION_DATABASE_NAME


# Configuration for Data Validation component
class DataValidaitonConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        # Root directory for all validation outputs
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_VALIDATION_DIR_NAME
        )

        # Directory containing valid datasets
        self.valid_data_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_VALID_DIR
        )

        # Directory containing invalid datasets
        self.invalid_data_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_INVALID_DIR
        )

        # Path to validated training dataset
        self.valid_training_file_path: str = os.path.join(
            self.valid_data_dir,
            training_pipeline.TRAIN_FILE_NAME
        )

        # Path to validated test dataset
        self.valid_test_file_path: str = os.path.join(
            self.valid_data_dir,
            training_pipeline.TEST_FILE_NAME
        )

        # Path where invalid training data is stored
        self.invalid_training_file_path: str = os.path.join(
            self.invalid_data_dir,
            training_pipeline.TRAIN_FILE_NAME
        )

        # Path where invalid test data is stored
        self.invalid_test_file_path: str = os.path.join(
            self.invalid_data_dir,
            training_pipeline.TEST_FILE_NAME
        )

        # Path to generated data drift report
        self.drift_file_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
            training_pipeline.DATA_VALIDATION_DRIFT_FILE_NAME
        )


# Configuration for Data Transformation component
class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        # Root directory for transformation artifacts
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_TRANSFORMATION_DIR_NAME
        )

        # Path to transformed training array (.npy)
        self.transformed_train_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TRAIN_FILE_NAME.replace("csv", "npy")
        )

        # Path to transformed test array (.npy)
        self.transformed_test_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TEST_FILE_NAME.replace("csv", "npy")
        )

        # Path to transformation object (scaler/encoder pipeline)
        self.transformed_object_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            training_pipeline.PREPROCESSING_OBJECT_FILE_NAME
        )


# Configuration for Model Trainer component
class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        # Directory for saving trained model and metrics
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.MODEL_TRAINER_DIR
        )

        # Path to final trained model file
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir,
            training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR,
            training_pipeline.MODEL_TRAINER_TRAINED_MODEL_NAME
        )

        # Minimum acceptable accuracy (below this training is considered failed)
        self.expected_accuracy: float = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE

        # Maximum allowed difference between train and test accuracy before labeling as overfitting/underfitting
        self.overfitting_underfitting_threshold: float = training_pipeline.MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD
