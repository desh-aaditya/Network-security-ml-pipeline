import os
import sys
import mlflow
import mlflow.sklearn

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models,
)
from networksecurity.utils.ml_utils.metric.classification_metric import (
    get_classification_metric as get_classification_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
import dagshub
dagshub.init(repo_owner='desh-aaditya', repo_name='Malicious-Website-Detection-using-Machine-Learning', mlflow=True)

class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, model, train_metric, test_metric):
        """
        Tracks metrics, parameters and model in MLflow
        """
        with mlflow.start_run():

            # ---- Train metrics ----
            mlflow.log_metric("train_f1_score", train_metric.f1_score)
            mlflow.log_metric("train_precision", train_metric.precision_score)
            mlflow.log_metric("train_recall", train_metric.recall_score)

            # ---- Test metrics ----
            mlflow.log_metric("test_f1_score", test_metric.f1_score)
            mlflow.log_metric("test_precision", test_metric.precision_score)
            mlflow.log_metric("test_recall", test_metric.recall_score)

            # ---- Model parameters ----
            mlflow.log_params(model.get_params())

            # ---- Log model ----
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
            )

    def train_model(self, X_train, y_train, X_test, y_test):
        try:
            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "AdaBoost": AdaBoostClassifier(),
            }

            params = {
                "Decision Tree": {
                    "criterion": ["gini", "entropy", "log_loss"],
                },
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 128, 256],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Logistic Regression": {},
                "AdaBoost": {
                    "learning_rate": [0.1, 0.01, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            # ---- Model evaluation ----
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
            )

            logging.info(f"Model evaluation report: {model_report}")

            # ---- Select best model (model_report values are FLOATS) ----
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            best_model = models[best_model_name]

            logging.info(
                f"Best model selected: {best_model_name} | Score: {best_model_score}"
            )

            # ---- Train & test metrics ----
            y_train_pred = best_model.predict(X_train)
            train_metric = get_classification_score(
                y_true=y_train, y_pred=y_train_pred
            )

            y_test_pred = best_model.predict(X_test)
            test_metric = get_classification_score(
                y_true=y_test, y_pred=y_test_pred
            )

            # ---- MLflow tracking ----
            self.track_mlflow(
                model=best_model,
                train_metric=train_metric,
                test_metric=test_metric,
            )

            # ---- Load preprocessor ----
            preprocessor = load_object(
                self.data_transformation_artifact.transformed_object_file_path
            )

            # ---- Save final model ----
            model_dir = os.path.dirname(
                self.model_trainer_config.trained_model_file_path
            )
            os.makedirs(model_dir, exist_ok=True)

            network_model = NetworkModel(
                preprocessor=preprocessor,
                model=best_model,
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=network_model,
            )
            save_object("final_models/model.pkl",best_model)
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric,
            )

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            return self.train_model(X_train, y_train, X_test, y_test)

        except Exception as e:
            raise NetworkSecurityException(e, sys)
