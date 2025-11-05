import yaml
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os, sys
import numpy as np
import dill
import pickle
from sklearn.model_selection import GridSearchCV
import sys
from typing import Dict, Tuple
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score



def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e

def write_yaml_file(file_path:str, content:object, replace:bool = False)-> None:
    try:

        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)

    except Exception as e:
        raise NetworkSecurityException(e, sys)
    

def save_numpy_arry_data(file_path:str, array: np.array):
    """
    Save the numpy array data to a file
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file=file_obj, arr=array)

    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
    
def load_numpy_array_data(file_path: str):
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file at this path {file_path} does not exist")
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    


def save_object(file_path: str, obj: object):
    try:
        logging.info("Entered the save object method in the mainUtils class")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(file=file_obj, obj=obj)

        logging.info("Exiting the save object method")
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def load_object(file_path: str):
    try: 
        if not os.path.exists(file_path):
            raise Exception(f"The {file_path} does not exist")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
import sys
from typing import Dict, Tuple
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from networksecurity.exception.exception import NetworkSecurityException

def _safe_scores_for_auc(fitted, X_test, is_binary: bool):
    """
    Returns score vector for ROC AUC when possible:
      - predict_proba[:,1] for binary, full proba for multiclass
      - decision_function as fallback
      - None if neither available
    """
    try:
        if hasattr(fitted, "predict_proba"):
            proba = fitted.predict_proba(X_test)
            return proba[:, 1] if is_binary else proba
        if hasattr(fitted, "decision_function"):
            return fitted.decision_function(X_test)
    except Exception:
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

    Returns:
      - report:          {model_name: test_f1_score}                # kept for backward compatibility
      - best_estimators: {model_name: fitted_best_estimator or None}
      - best_params:     {model_name: best_params_dict or {}}
      - scores:          {model_name: {"f1": ..., "accuracy": ..., "roc_auc": ...}}
    """
    try:
        report: Dict[str, float] = {}
        best_estimators: Dict[str, object] = {}
        best_params: Dict[str, dict] = {}
        scores: Dict[str, Dict[str, float]] = {}

        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        is_binary = (len(np.unique(y_train)) == 2)

        for name, model in models.items():
            try:
                grid = params.get(name, {}) or {}

                # Tune (if grid provided) or fit with defaults
                if len(grid) > 0:
                    gs = GridSearchCV(
                        estimator=model,
                        param_grid=grid,
                        cv=cv,
                        n_jobs=n_jobs,
                        verbose=verbose
                        # NOTE: no 'scoring' passed per your request
                    )
                    gs.fit(X_train, y_train)
                    fitted = gs.best_estimator_
                    best_params[name] = gs.best_params_
                else:
                    fitted = model.fit(X_train, y_train)
                    best_params[name] = {}

                # --- Evaluate on test set (multi-metric) ---
                y_pred = fitted.predict(X_test)

                # F1: binary → 'binary', otherwise → 'weighted'
                f1_avg = "binary" if is_binary else "weighted"
                f1_val = f1_score(y_test, y_pred, average=f1_avg)

                acc_val = accuracy_score(y_test, y_pred)

                auc_val = None
                y_scores = _safe_scores_for_auc(fitted, X_test, is_binary=is_binary)
                if y_scores is not None:
                    try:
                        if is_binary:
                            auc_val = roc_auc_score(y_test, y_scores)
                        else:
                            auc_val = roc_auc_score(y_test, y_scores, multi_class="ovr")
                    except Exception:
                        auc_val = None

                # record per-model metrics
                scores[name] = {
                    "f1": float(f1_val),
                    "accuracy": float(acc_val),
                    "roc_auc": (float(auc_val) if auc_val is not None else float("nan"))
                }

                # keep report as F1 only (so your winner-pick code still works)
                report[name] = float(f1_val)
                best_estimators[name] = fitted

            except Exception:
                report[name] = float("nan")
                best_estimators[name] = None
                best_params[name] = {}
                scores[name] = {
                    "f1": float("nan"),
                    "accuracy": float("nan"),
                    "roc_auc": float("nan")
                }

        return report, best_estimators, best_params, scores

    except Exception as e:
        raise NetworkSecurityException(e, sys)