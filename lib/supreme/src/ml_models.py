import logging
from typing import Dict, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from ray import tune
from settings import X_TIME
from sklearn.cluster import (
    DBSCAN,
    AffinityPropagation,
    AgglomerativeClustering,
    KMeans,
    MeanShift,
    SpectralClustering,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    completeness_score,
    f1_score,
    homogeneity_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from tune_sklearn import TuneSearchCV
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class MLModels:
    def __init__(self, model: str, x_train: pd.DataFrame, y_train: pd.DataFrame):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train

    def train_ml_model_factory(self):
        models = {
            "MLP": self.MLP,
            "XGB": self.XGB,
            "LGBM": self.LGBM,
            "RF": self.RF,
            "SVM": self.SVM,
            "LR": self.LR,
            "SGD": self.SGD,
        }
        try:
            return models.get(self.model)()
        except ValueError:
            raise Exception

    def train_classifier(self) -> Tuple:
        """
        Train a classifier

        """
        try:
            classfier = self.train_ml_model_factory()
            return classfier.fit(self.x_train, self.y_train)  # .best_estimator_

        except (TypeError, ValueError):
            return None

    def MLP(self):
        params = {
            "hidden_layer_sizes": [
                (16,),
                (32,),
                (64,),
                (128,),
                (256,),
                (512,),
                (32, 32),
                (64, 32),
                (128, 32),
                (256, 32),
                (512, 32),
            ],
            "solver": ["adam"],
            "activation": ["relu", "sgd"],
        }
        search = TuneSearchCV(
            estimator=MLPClassifier(),
            return_train_score=True,
            scoring="f1_macro",
            param_distributions=params,
            cv=4,
            early_stopping=True,
            n_trials=X_TIME,
            verbose=0,
            random_state=0,
        )
        search.fit(self.x_train, self.y_train)
        return search.best_estimator

    def XGB(self):
        fit_params = {
            "reg_alpha": tune.choice([0, 1, 2, 3, 4, 5, 6]),
            "reg_lambda": tune.choice([0, 1, 2, 3, 4, 5]),
            "learning_rate": tune.choice([0, 0.01, 0.001, 0.0001, 0.02, 0.002, 0.0002]),
            "tree_method": ["approx"],
            "objective": ["binary:logistic"],
            "eval_metric": ["logloss", "error"],
            "eta": tune.loguniform(1e-4, 1e-1),
            "subsample": tune.uniform(0.5, 1.0),
            "max_depth": tune.choice([30, 50, 70, 90, 100, 150]),
            "n_estimators": tune.choice(
                [int(x) for x in np.linspace(start=50, stop=400, num=50)]
            ),
        }
        search = TuneSearchCV(
            estimator=XGBClassifier(),
            return_train_score=True,
            scoring="f1_macro",
            param_distributions=fit_params,
            cv=4,
            n_trials=X_TIME,
            search_optimization="hyperopt",
            early_stopping=True,
            verbose=0,
            random_state=0,
        )

        search.fit(self.x_train, self.y_train)

        return search.best_estimator

    def LGBM(self):
        param_dists = {
            "n_estimators": tune.choice(
                [int(x) for x in np.linspace(start=50, stop=400, num=50)]
            ),
            "colsample_bytree": tune.choice([0.7, 0.8]),
            "max_depth": tune.choice([1, 5, 7]),
            "num_leaves": tune.choice([5, 7, 9]),
            "reg_alpha": tune.choice([1.1, 1.2, 1.3]),
            "reg_lambda": tune.choice([1.1, 1.2, 1.3]),
            "min_split_gain": tune.choice([0.3, 0.4]),
            "subsample": tune.choice([0.7, 0.8, 0.9]),
            "subsample_freq": tune.choice([3]),
            "boosting_type": tune.choice(["gbdt", "dart"]),
            "learning_rate": tune.choice([0.005, 0.01]),
        }
        search = TuneSearchCV(
            lgb.LGBMClassifier(),
            return_train_score=True,
            scoring="f1_macro",
            param_distributions=param_dists,
            cv=4,
            n_trails=X_TIME,
            n_jobs=-1,
            early_stopping=True,
            search_optimization="optuna",
            verbose=0,
        )

        search.fit(self.x_train, self.y_train)

        return search.best_estimator

    def RF(self):
        max_depth = [int(x) for x in np.linspace(50, 300, num=50)]
        max_depth.append(None)
        params = {
            "n_estimators": [int(x) for x in np.linspace(start=50, stop=400, num=50)]
        }
        search = TuneSearchCV(
            estimator=RandomForestClassifier(),
            return_train_score=True,
            scoring="f1_macro",
            param_distributions=params,
            cv=4,
            n_iter=X_TIME,
            search_optimization="optuna",
            early_stopping=True,
            verbose=0,
        )
        search.fit(self.x_train, self.y_train)
        return search.best_estimator

    def SVM(self):
        params = {
            "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "gamma": [1, 0.1, 0.01, 0.001],
        }
        search = TuneSearchCV(
            SVC(),
            return_train_score=True,
            scoring="f1_macro",
            param_distributions=params,
            cv=4,
            n_iter=X_TIME,
            search_optimization="optuna",
            early_stopping=True,
            verbose=0,
        )
        search.fit(self.x_train, self.y_train)
        return search.best_estimator

    def SGD(self):
        param_dists = {
            "loss": ["squared_hinge", "hinge"],
            "alpha": [0.01, 0.001, 0.0001],
            "epsilon": [0.01, 0.001],
        }
        search = TuneSearchCV(
            SGDClassifier(),
            return_train_score=True,
            scoring="f1_macro",
            param_distributions=param_dists,
            cv=4,
            n_iter=X_TIME,
            early_stopping=True,
            search_optimization="optuna",
            verbose=0,
        )
        search.fit(self.x_train, self.y_train)
        return search.best_estimator

    def LR(self):
        LRmodel = LogisticRegression(max_iter=300)
        logic_grid = {
            "solver": ["liblinear", "sag", "saga"],
            "penalty": ["l1", "l2"],  # , "elasticnet"],
            "class_weight": ["balanced"],
            "C": [1, 10, 100],
        }
        search = TuneSearchCV(
            LRmodel(),
            return_train_score=True,
            scoring="f1_macro",
            param_distributions=logic_grid,
            cv=4,
            n_iter=X_TIME,
            search_optimization="optuna",
            early_stopping=True,
            verbose=0,
        )
        search.fit(self.x_train, self.y_train)
        return search.best_estimator

    def get_result(
        self, model, results: Dict, X_test: np.ndarray, y_test: np.ndarray
    ) -> None:
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(X_test)

        train_pred = model.predict(self.x_train)

        results["test_accuracy"].append(round(accuracy_score(y_test, y_pred), 3))
        results["test_weighted_f1"].append(
            round(f1_score(y_test, y_pred, average="weighted"), 3)
        )
        results["test_macro_f1"].append(
            round(f1_score(y_test, y_pred, average="macro"), 3)
        )
        results["train_accuracy"].append(
            round(accuracy_score(self.y_train, train_pred), 3)
        )
        results["train_weighted_f1"].append(
            round(f1_score(self.y_train, train_pred, average="weighted"), 3)
        )
        results["train_macro_f1"].append(
            round(f1_score(self.y_train, train_pred, average="macro"), 3)
        )


class ClusteringModels:
    def __init__(
        self, model: str, x_train: pd.DataFrame, y_train: Optional[pd.DataFrame] = None
    ):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train

    def train_ml_model_factory(self):
        models = {
            "KM": self.KM,
            "APF": self.AFP,
            "DBSCAN": self.DBSCAN,
            "AGC": self.AGC,
            "MS": self.MS,
            "SPC": self.SPC,
        }
        try:
            return models.get(self.model)()
        except ValueError:
            raise Exception

    def train_classifier(self) -> Tuple:
        """
        Train a classifier

        """
        try:
            classfier = self.train_ml_model_factory()
            return classfier.fit(self.x_train, self.y_train)  # .best_estimator_

        except (TypeError, ValueError):
            return None

    def KM(self):
        params = {
            "n_clusters": [7, 8, 9, 10, 11],
            "init": ["k-means++"],
            "n_init": [10, 20, 30],
        }

        search = TuneSearchCV(
            KMeans(),
            param_distributions=params,
            cv=4,
            n_iter=X_TIME,
            scoring=silhouette_score,
            verbose=0,
        )

        search.fit(self.x_train)

        # best_n_clusters = search.best_params_["n_clusters"]
        # best_init = search.best_params_["init"]
        # best_n_init = search.best_params_["n_init"]

        # kmeans_model = KMeans(
        #     n_clusters=best_n_clusters, init=best_init, n_init=best_n_init
        # )

        # return kmeans_model.fit(self.x_train), search
        return search.best_estimator
        # best_k = 7
        # sil_score = 0.0
        # for n_clusters in [7, 8, 9]:
        #     km = KMeans(
        #         n_clusters=n_clusters, init="k-means++", n_init="auto", max_iter=100
        #     )
        #     score = silhouette_score(self.x_train, km.fit_predict(self.x_train))
        #     if score > sil_score:
        #         sil_score = score
        #         best_k = n_clusters
        #     print(
        #         "For n_clusters =",
        #         n_clusters,
        #         "The average silhouette_score is :",
        #         score,
        #     )
        # print("-------------------------")
        # km = KMeans(n_clusters=best_k, init="k-means++", n_init="auto")
        # return km.fit(self.x_train), ""

    def AFP(self):
        param = {
            "damping": np.arange(0.1, 0.8, 0.1),
            "affinity": ["precomputed", "euclidean"],
        }

        def custome_score(estimator, x_train):
            return silhouette_score(estimator.fit_predict(x_train), x_train)

        search = TuneSearchCV(
            AffinityPropagation(),
            param_distributions=param,
            scoring=custome_score,
            cv=4,
            n_iter=X_TIME,
            verbose=0,
        )
        search.fit(self.x_train)
        return search.best_estimator

    def get_result(
        self,
        model,
        results: Dict,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> None:
        try:
            predictions = model.predict(self.x_train)
        except AttributeError:
            predictions = model.labels_
        results["homogeneity"].append(homogeneity_score(self.y_train, predictions))
        results["Completeness"].append(completeness_score(self.y_train, predictions))
        results["v_measure"].append(v_measure_score(self.y_train, predictions))
        results["adjusted_rand"].append(adjusted_rand_score(self.y_train, predictions))
        results["silhouette"].append(silhouette_score(self.x_train, predictions))

    def DBSCAN(self):
        dbscan_model = DBSCAN(eps=0.5, min_samples=5)
        param = {
            "eps": np.arange(0.1, 0.7, 0.1),
            "min_samples": [1, 5, 10, 15],
        }
        search = TuneSearchCV(
            dbscan_model, param_distributions=param, n_iter=X_TIME, verbose=0
        )
        search.fit(self.x_train)
        return search.best_estimator

    def AGC(self):
        agc = AgglomerativeClustering(
            n_clusters=5, affinity="euclidean", linkage="ward"
        )
        return agc.fit(self.x_train), "Add_best_later"

    def MS(self):
        """Add TuneSearchCV later"""
        return MeanShift().fit(self.x_train), "Add_best_later"

    def SPC(self):
        return SpectralClustering(n_clusters=5).fit(self.x_train), "Add_best_later"
