import os, sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import ModelTrainerArtifact, ClassificationMetricArtifact, DataTransformationArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.main_utils.utils import save_object, load_numpy_array_data, load_object, evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.constant.training_pipeline import REPO_NAME, REPO_OWNER
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
import mlflow
import dagshub

dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def track_model(self, best_model, classification_metric_artifact:ClassificationMetricArtifact):
        try:
            with mlflow.start_run():
                f1_score = classification_metric_artifact.f1_score
                accuracy_score = classification_metric_artifact.accuracy_score
                precision_score = classification_metric_artifact.precision_score
                recall_score = classification_metric_artifact.recall_score

                mlflow.log_metric("f1_score", f1_score)
                mlflow.log_metric("accuracy_score", accuracy_score)
                mlflow.log_metric("precision_score", precision_score)
                mlflow.log_metric("recall_score", recall_score)

                mlflow.sklearn.log_model(best_model, "model")

        except Exception as e:
            raise NetworkSecurityException(e, sys)

        
    def train_model(self, X_train, y_train, X_test, y_test):
        # ----- 1) Define models -----
        models = {
            "LogisticRegression": LogisticRegression(verbose=1, n_jobs=None),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(verbose=1),
            "RandomForestClassifier": RandomForestClassifier(verbose=1, random_state=42),
            "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
            "AdaBoostClassifier": AdaBoostClassifier(random_state=42)
        }

        # ----- 2) Param grids -----
        params = {
            "LogisticRegression": {
                "penalty": ["l2", None],
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["lbfgs", "saga"],
                "max_iter": [500],
                "class_weight": [None, "balanced"]
            },
            "KNeighborsClassifier": {
                "n_neighbors": [3, 5, 7, 11],
                "weights": ["uniform", "distance"],
                "metric": ["minkowski"],
                "p": [1, 2]
            }
            # ,

            # "GradientBoostingClassifier": {
            #     "n_estimators": [100, 200, 400],
            #     "learning_rate": [0.05, 0.1],
            #     "max_depth": [2, 3, 4],
            #     "subsample": [0.9, 1.0],
            #     "min_samples_split": [2, 5],
            #     "min_samples_leaf": [1, 2]
            # }
            # ,
            # "RandomForestClassifier": {
            #     "n_estimators": [200, 400],
            #     "max_depth": [None, 20, 30],
            #     "min_samples_split": [2, 5],
            #     "min_samples_leaf": [1, 2],
            #     "max_features": ["sqrt", "log2"],
            #     "bootstrap": [True],
            #     "class_weight": [None, "balanced_subsample"]
            # },
            # "DecisionTreeClassifier": {
            #     "criterion": ["gini", "entropy"],
            #     "max_depth": [None, 10, 20],
            #     "min_samples_split": [2, 5],
            #     "min_samples_leaf": [1, 2],
            #     "class_weight": [None, "balanced"]
            # },
            # "AdaBoostClassifier": {
            #     "n_estimators": [100, 200, 400],
            #     "learning_rate": [0.05, 0.1, 1.0],
            #     "algorithm": ["SAMME.R"]
            # }
        }

        # ----- 3) Evaluate all models -----
        report, best_estimators, best_params, scores = evaluate_models(
            X_train, y_train, X_test, y_test,
            models=models,
            params=params,
            cv_splits=5,
            n_jobs=-1,
            verbose=1
        )

        # ----- 3b) Log each model’s best params + metric bundle BEFORE picking winner -----
        logging.info("==== Per-model summary (best params + metrics) ====")
        for model_name in models.keys():
            bp = best_params.get(model_name, {})
            sc = scores.get(model_name, {})
            # sc expected shape: {"f1": ..., "accuracy": ..., "roc_auc": ...}
            logging.info(
                "Model: %s | best_params=%s | metrics={f1: %.6f, accuracy: %.6f, roc_auc: %s}",
                model_name,
                bp,
                float(sc.get("f1", float("nan"))),
                float(sc.get("accuracy", float("nan"))),
                ("%.6f" % sc["roc_auc"]) if (sc.get("roc_auc") is not None and sc.get("roc_auc") == sc.get("roc_auc")) else "nan"
            )



        # ----- 4) Pick the winner -----
        valid_scores = {}
        for model_name, score in report.items():
            if score == score:  # filter NaN
                valid_scores[model_name] = score

        if not valid_scores:
            raise NetworkSecurityException("All model evaluations failed; report is empty/NaN.", sys)

        best_model_name = max(valid_scores, key=valid_scores.get)
        best_model_score = valid_scores[best_model_name]
        best_model = best_estimators[best_model_name]

        logging.info(
            "Best model: %s | F1=%.4f | params=%s | full_metrics=%s",
            best_model_name, best_model_score, best_params.get(best_model_name),
            scores.get(best_model_name)
        )

        # ----- 5) Compute train/test classification metrics (your helper) -----
        y_train_pred = best_model.predict(X_train)
        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

        y_test_pred = best_model.predict(X_test)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

        # ----- 6) Wrap with preprocessor and persist -----
        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=network_model)

        ##Saving the final model
        save_object(file_path="final_model/model.pkl", obj=best_model)


        ##tracking with MLFlow
        self.track_model(best_model=best_model, classification_metric_artifact=classification_train_metric)
        self.track_model(best_model=best_model, classification_metric_artifact=classification_test_metric)

        # ----- 7) Build and return artifact -----
        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric
        )
        return model_trainer_artifact




        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train, X_test, y_test = (train_arr[:,:-1], 
                                                train_arr[:, -1], 
                                                test_arr[:, :-1], 
                                                test_arr[:, -1])

            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    


