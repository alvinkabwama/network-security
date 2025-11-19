from networksecurity.entity.artifact_entity import ClassificationMetricArtifact  # Dataclass to hold classification metrics
from networksecurity.exception.exception import NetworkSecurityException  # Custom exception used across the project
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score  # sklearn metric functions
import os, sys  # OS utilities (unused here) and sys for passing exception details


def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    """
    Compute standard classification metrics (F1, precision, recall, accuracy)
    and return them wrapped in a ClassificationMetricArtifact.
    """
    try:
        # Compute F1 score (default: binary or 'binary' average depending on labels)
        model_f1_score = f1_score(y_true=y_true, y_pred=y_pred)

        # Compute recall score (true positive rate)
        model_recall_score = recall_score(y_true=y_true, y_pred=y_pred)

        # Compute precision score (how many predicted positives are correct)
        model_precision_score = precision_score(y_true=y_true, y_pred=y_pred)

        # Compute overall accuracy (percentage of correct predictions)
        model_accuracy_score = accuracy_score(y_true=y_true, y_pred=y_pred)

        # Package all metrics into a ClassificationMetricArtifact dataclass
        classification_metric = ClassificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score,
            recall_score=model_recall_score,
            accuracy_score=model_accuracy_score
        )
        
        # Return metrics object to the caller
        return classification_metric
    
    except Exception as e:
        # Wrap any metric computation error into the custom project exception
        raise NetworkSecurityException(e, sys)
