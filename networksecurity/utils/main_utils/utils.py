import yaml  # For reading and writing YAML configuration files
from networksecurity.exception.exception import NetworkSecurityException  # Custom project exception
from networksecurity.logging.logger import logging  # Project logger
import os, sys  # OS utilities and access to exception info
import numpy as np  # Numerical arrays and saving/loading .npy
import dill  # (Imported but not used here) - can serialize complex Python objects
import pickle  # For serializing Python objects to disk
from sklearn.model_selection import GridSearchCV  # Hyperparameter tuning via grid search
import sys  # Re-imported (already imported above)
from typing import Dict, Tuple  # Type hints for dictionaries and tuples
import numpy as np  # Re-imported numpy
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # Stratified K-fold CV for classification
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score  # Evaluation metrics


def read_yaml_file(file_path: str) -> dict:
    """
    Read and parse a YAML file into a Python dictionary.
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        # Wrap any YAML/IO error into custom exception for consistent handling
        raise NetworkSecurityException(e, sys) from e

def write_yaml_file(file_path:str, content:object, replace:bool = False)-> None:
    """
    Write a Python object (usually dict) to a YAML file.

    :param file_path: Destination YAML file path
    :param content:   Data to serialize and write
    :param replace:   If True, delete existing file before writing
    """
    try:

        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)  # Remove old file if replace is requested

        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Write YAML content to file
        with open(file_path, "w") as file:
            yaml.dump(content, file)

    except Exception as e:
        # Wrap exceptions (permissions, IO, etc.)
        raise NetworkSecurityException(e, sys)
    

def save_numpy_arry_data(file_path:str, array: np.array):
    """
    Save the numpy array data to a file in .npy format.
    """
    try:
        # Ensure target directory exists
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save numpy array to binary file
        with open(file_path, "wb") as file_obj:
            np.save(file=file_obj, arr=array)

    except Exception as e:
        # Wrap exception with project-specific context
        raise NetworkSecurityException(e, sys)
    
    
def load_numpy_array_data(file_path: str):
    """
    Load a numpy array from a given .npy file path.
    """
    try:
        # Check if file exists before loading
        if not os.path.exists(file_path):
            raise Exception(f"The file at this path {file_path} does not exist")

        # Load numpy array from file
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        # Wrap any IO/numpy errors
        raise NetworkSecurityException(e, sys)
    


def save_object(file_path: str, obj: object):
    """
    Serialize and save a Python object using pickle.
    """
    try:
        logging.info("Entered the save object method in the mainUtils class")

        # Ensure destination directory exists
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Serialize object to file via pickle
        with open(file_path, "wb") as file_obj:
            pickle.dump(file=file_obj, obj=obj)

        logging.info("Exiting the save object method")
    except Exception as e:
        # Wrap any exceptions from directory creation or pickling
        raise NetworkSecurityException(e, sys)
    
def load_object(file_path: str):
    """
    Load and deserialize a Python object from a pickle file.
    """
    try: 
        # Ensure file exists
        if not os.path.exists(file_path):
            raise Exception(f"The {file_path} does not exist")

        # Load pickled object from file
        with open(file_path, "rb") as file_obj:
            print(file_obj)  # Debug print of file object reference
            return pickle.load(file_obj)
    except Exception as e:
        # Wrap exceptions for unified error handling
        raise NetworkSecurityException(e, sys)
    

import sys  # Duplicate import (kept as-is intentionally)
from typing import Dict, Tuple  # Duplicate import (kept as-is intentionally)
import numpy as np  # Duplicate import (kept as-is intentionally)
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # Duplicate import (kept as-is intentionally)
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score  # Duplicate import (kept as-is intentionally)
from networksecurity.exception.exception import NetworkSecurityException  # Duplicate import (kept as-is intentionally)

def _safe_scores_for_auc(fitted, X_test, is_binary: bool):
    """
    Safely obtain score vector for ROC AUC when possible:

      - For models with predict_proba:
          - return proba[:, 1] for binary
          - return full probability matrix for multiclass
      - If predict_proba is unavailable, use decision_function when present.
      - If neither is available or fails, return None.
    """
    try:
        if hasattr(fitted, "predict_proba"):
            proba = fitted.predict_proba(X_test)
            return proba[:, 1] if is_binary else proba
        if hasattr(fitted, "decision_function"):
            return fitted.decision_function(X_test)
    except Exception:
        # Silently ignore errors and fall back to None
        pass
    return None


def evaluate_models(
    X_train, y_train, X_test, y_test,
    models: Dict[str, object],
    params: Dict[str, dict],
    cv_splits: int = 5,
    n_jobs: int = -1,
    verbose: int = 0
) -> Tuple[Dict[str, float], Dict[str, object], Dict[str, dict], Dict[str, Dict[str, float]]]:
    """
    Train/tune multiple classifiers and evaluate on test set.

    Parameters
    ----------
    X_train, y_train : Training features and labels
    X_test, y_test   : Test features and labels
    models           : Dict of model_name -> estimator instance
    params           : Dict of model_name -> param_grid for GridSearchCV
    cv_splits        : Number of StratifiedKFold splits
    n_jobs           : Parallel jobs for GridSearchCV
    verbose          : Verbosity level for GridSearchCV

    Returns
    -------
      - report:
          {model_name: test_f1_score}
          (kept for backward compatibility as main selection metric)
      - best_estimators:
          {model_name: fitted_best_estimator or None}
      - best_params:
          {model_name: best_params_dict or {}}
      - scores:
          {
            model_name: {
              "f1": ...,
              "accuracy": ...,
              "roc_auc": ...
            }
          }
    """
    try:
        # Primary F1-only report (legacy usage)
        report: Dict[str, float] = {}
        # Fitted best models per algorithm
        best_estimators: Dict[str, object] = {}
        # Best hyperparameters found by grid search per model
        best_params: Dict[str, dict] = {}
        # Detailed metric scores per model
        scores: Dict[str, Dict[str, float]] = {}

        # Stratified K-fold to preserve class balance in each split
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

        # Check if problem is binary classification
        is_binary = (len(np.unique(y_train)) == 2)

        # Loop through each model and its associated hyperparameter grid
        for name, model in models.items():
            try:
                grid = params.get(name, {}) or {}

                # If param grid is provided, use GridSearchCV, else fit with default params
                if len(grid) > 0:
                    gs = GridSearchCV(
                        estimator=model,
                        param_grid=grid,
                        cv=cv,
                        n_jobs=n_jobs,
                        verbose=verbose
                        # NOTE: no explicit 'scoring' passed - uses default estimator score
                    )
                    gs.fit(X_train, y_train)
                    fitted = gs.best_estimator_  # Best model from grid search
                    best_params[name] = gs.best_params_
                else:
                    # No grid: train the model directly with default params
                    fitted = model.fit(X_train, y_train)
                    best_params[name] = {}

                # --- Evaluate on test set (multi-metric) ---
                y_pred = fitted.predict(X_test)

                # F1 averaging mode:
                #   - 'binary' for binary classification
                #   - 'weighted' for multiclass
                f1_avg = "binary" if is_binary else "weighted"
                f1_val = f1_score(y_test, y_pred, average=f1_avg)

                # Standard accuracy
                acc_val = accuracy_score(y_test, y_pred)

                # ROC AUC (if possible)
                auc_val = None
                y_scores = _safe_scores_for_auc(fitted, X_test, is_binary=is_binary)
                if y_scores is not None:
                    try:
                        if is_binary:
                            auc_val = roc_auc_score(y_test, y_scores)
                        else:
                            # Multiclass AUC using one-vs-rest
                            auc_val = roc_auc_score(y_test, y_scores, multi_class="ovr")
                    except Exception:
                        # If ROC-AUC computation fails, keep as None
                        auc_val = None

                # Store detailed metrics for this model
                scores[name] = {
                    "f1": float(f1_val),
                    "accuracy": float(acc_val),
                    "roc_auc": (float(auc_val) if auc_val is not None else float("nan"))
                }

                # Maintain backward-compatible report keyed by F1 score only
                report[name] = float(f1_val)
                best_estimators[name] = fitted

            except Exception:
                # If something fails for a specific model, mark metrics as NaN and estimator as None
                report[name] = float("nan")
                best_estimators[name] = None
                best_params[name] = {}
                scores[name] = {
                    "f1": float("nan"),
                    "accuracy": float("nan"),
                    "roc_auc": float("nan")
                }

        # Return all aggregated evaluation artifacts
        return report, best_estimators, best_params, scores

    except Exception as e:
        # Wrap top-level evaluation errors with project-specific exception
        raise NetworkSecurityException(e, sys)
