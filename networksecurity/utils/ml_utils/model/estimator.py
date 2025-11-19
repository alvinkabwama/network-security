from networksecurity.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME  # Constants for saved model directory and filename

import os, sys  # OS utilities (unused here) and sys for passing exception details
from networksecurity.exception.exception import NetworkSecurityException  # Custom exception wrapper
from networksecurity.logging.logger import logging  # Project logger (currently unused in this class)


class NetworkModel:
    """
    Wrapper model that combines a preprocessor and a trained model.
    This ensures that any input data is transformed using the same
    preprocessing pipeline before predictions are made.
    """
    def __init__(self, preprocessor, model):
        try:
            # Store the preprocessing pipeline (e.g., scaler, encoder, etc.)
            self.preprocessor = preprocessor
            # Store the trained model (classifier/regressor)
            self.model = model
        except Exception as e:
            # Wrap any initialization error in the custom exception
            raise NetworkSecurityException(e, sys)
        
    def predict(self, x):
        """
        Apply preprocessing to the input 'x' and predict using the wrapped model.
        """
        try:
            # Transform incoming raw features using the stored preprocessor
            x_transform = self.preprocessor.transform(x)
            # Use the trained model to generate predictions on transformed data
            y_hat = self.model.predict(x_transform)
            # Return predictions to the caller
            return (y_hat)
        
        except Exception as e:
            # Wrap prediction-time errors (e.g., shape mismatch) in custom exception
            raise NetworkSecurityException(e, sys)
