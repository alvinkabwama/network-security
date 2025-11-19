from dataclasses import dataclass  # Used to automatically generate init, repr, and other methods for simple data containers


# Artifact produced after data ingestion, containing paths to the processed training and test files
@dataclass
class DataIngestionArtifact:
    training_file_path: str   # Path to the final training dataset file
    test_file_path: str       # Path to the final test dataset file


# Artifact produced after data validation, containing validation status and file paths for valid/invalid datasets
@dataclass
class DataValidationArtifact:
    validation_status: bool            # Indicates whether validation passed or failed
    valid_train_file_path: str         # Path to the validated training dataset
    valid_test_file_path: str          # Path to the validated test dataset
    invalid_train_file_path: str       # Path where invalid training rows are stored
    invalid_test_file_path: str        # Path where invalid test rows are stored
    drift_report_file_path: str        # Path to the data drift report generated during validation


# Artifact produced after data transformation, containing transformed object and transformed datasets
@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str  # Path to the serialized transformation pipeline (scaler, encoder, etc.)
    transformed_train_file_path: str   # Path to the transformed training dataset array
    transformed_test_file_path: str    # Path to the transformed test dataset array


# Evaluation metrics returned after model evaluation (train or test)
@dataclass
class ClassificationMetricArtifact:
    f1_score: float        # F1 Score: harmonic mean of precision and recall
    precision_score: float # Precision metric
    recall_score: float    # Recall metric
    accuracy_score: float  # Accuracy of the classifier


# Artifact produced after model training, containing model file path and evaluation metrics
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str                 # Path where the trained model is saved
    train_metric_artifact: ClassificationMetricArtifact  # Metrics on the training dataset
    test_metric_artifact: ClassificationMetricArtifact   # Metrics on the test dataset
