import logging
from typing import Dict, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
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
            ]
        }
        search = RandomizedSearchCV(
            estimator=MLPClassifier(
                solver="adam", activation="relu", early_stopping=True
            ),
            return_train_score=True,
            scoring="f1_macro",
            param_distributions=params,
            cv=4,
            n_iter=X_TIME,
            verbose=0,
        )
        search.fit(self.x_train, self.y_train)
        return (
            MLPClassifier(
                solver="adam",
                activation="relu",
                early_stopping=True,
                hidden_layer_sizes=search.best_params_["hidden_layer_sizes"],
            ),
            search,
        )

    def XGB(self):
        params = {
            "reg_alpha": range(0, 6, 1),
            "reg_lambda": range(1, 5, 1),
            "learning_rate": [0, 0.001, 0.01, 1],
        }
        fit_params = {
            "early_stopping_rounds": 10,
            "eval_metric": "mlogloss",
            "eval_set": [(self.x_train, self.y_train)],
        }

        search = RandomizedSearchCV(
            estimator=XGBClassifier(
                use_label_encoder=False,
                n_estimators=1000,
                fit_params=fit_params,
                objective="multi:softprob",
                eval_metric="mlogloss",
                verbosity=0,
            ),
            return_train_score=True,
            scoring="f1_macro",
            param_distributions=params,
            cv=4,
            n_iter=X_TIME,
            verbose=0,
        )

        search.fit(self.x_train, self.y_train)

        return (
            XGBClassifier(
                use_label_encoder=False,
                objective="multi:softprob",
                eval_metric="mlogloss",
                verbosity=0,
                n_estimators=1000,
                fit_params=fit_params,
                reg_alpha=search.best_params_["reg_alpha"],
                reg_lambda=search.best_params_["reg_lambda"],
                learning_rate=search.best_params_["learning_rate"],
            ),
            search,
        )

    def LGBM(self):
        param_dists = {
            "n_estimators": [int(x) for x in np.linspace(start=1, stop=5, num=1)],
            "colsample_bytree": [0.7, 0.8],
            "max_depth": [1, 5, 7],
            "num_leaves": [5, 7, 9],
            "reg_alpha": [1.1, 1.2, 1.3],
            "reg_lambda": [1.1, 1.2, 1.3],
            "min_split_gain": [0.3, 0.4],
            "subsample": [0.7, 0.8, 0.9],
            "subsample_freq": [3],
            "boosting_type": ["gbdt", "dart"],
            "learning_rate": [0.005, 0.01],
        }
        search = RandomizedSearchCV(
            estimator=lgb.LGBMClassifier(
                use_label_encoder=False,
                n_estimators=1000,
                fit_params=param_dists,
                objective="multi:softprob",
                eval_metric="mlogloss",
                verbosity=0,
            ),
            return_train_score=True,
            scoring="f1_macro",
            param_distributions=param_dists,
            cv=4,
            n_iter=X_TIME,
            verbose=0,
        )

        search.fit(self.x_train, self.y_train)

        return (
            lgb.LGBMClassifier(
                use_label_encoder=False,
                objective="multi:softprob",
                eval_metric="mlogloss",
                verbosity=0,
                n_estimators=1000,
                fit_params=param_dists,
                reg_alpha=search.best_params_["reg_alpha"],
                reg_lambda=search.best_params_["reg_lambda"],
                learning_rate=search.best_params_["learning_rate"],
            ),
            search,
        )

    def RF(self):
        # max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        # max_depth.append(None)
        # params = {
        #     "n_estimators": [int(x) for x in np.linspace(start=200, stop=2000, num=100)]
        # }
        # search = RandomizedSearchCV(
        #     estimator=RandomForestClassifier(),
        #     return_train_score=True,
        #     scoring="f1_macro",
        #     param_distributions=params,
        #     cv=4,
        #     n_iter=X_TIME,
        #     verbose=0,
        # )
        # search.fit(self.x_train, self.y_train)
        return RandomForestClassifier(n_estimators=10)  # ,
        # search,
        # )

    def SVM(self):
        params = {
            "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "gamma": [1, 0.1, 0.01, 0.001],
        }
        search = RandomizedSearchCV(
            SVC(),
            return_train_score=True,
            scoring="f1_macro",
            param_distributions=params,
            cv=4,
            n_iter=X_TIME,
            verbose=0,
        )
        search.fit(self.x_train, self.y_train)
        return (
            SVC(C=search.best_params_["C"], gamma=search.best_params_["gamma"]),
            search,
        )

    def SGD(self):
        param_dists = {
            "loss": ["squared_hinge", "hinge"],
            "alpha": [0.01, 0.001, 0.0001],
            "epsilon": [0.01, 0.001],
        }
        search = RandomizedSearchCV(
            SGDClassifier(),
            return_train_score=True,
            scoring="f1_macro",
            param_distributions=param_dists,
            cv=4,
            n_iter=X_TIME,
            verbose=0,
        )
        search.fit(self.x_train, self.y_train)
        return (
            SGDClassifier(),
            search,
        )

    def LR(self):
        LRmodel = LogisticRegression(max_iter=300)
        logic_grid = {
            "solver": ["liblinear", "sag", "saga"],
            "penalty": ["l1", "l2"],  # , "elasticnet"],
            "class_weight": ["balanced"],
            "C": [1, 10, 100],
        }
        search = RandomizedSearchCV(
            LRmodel(),
            return_train_score=True,
            scoring="f1_macro",
            param_distributions=logic_grid,
            cv=4,
            n_iter=X_TIME,
            verbose=0,
        )
        search.fit(self.x_train, self.y_train)
        return (
            LRmodel(),
            search,
        )

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
            "n_clusters": [5],
            "init": ["k-means++", "random"],
            "n_init": [10, 20, 30],
        }

        search = RandomizedSearchCV(
            KMeans(),
            param_distributions=params,
            cv=4,
            n_iter=X_TIME,
            verbose=0,
        )

        search.fit(self.x_train)

        best_n_clusters = search.best_params_["n_clusters"]
        best_init = search.best_params_["init"]
        best_n_init = search.best_params_["n_init"]

        kmeans_model = KMeans(
            n_clusters=best_n_clusters, init=best_init, n_init=best_n_init
        )

        return kmeans_model.fit(self.x_train), search

    def AFP(self):
        # param = {
        #     "damping" : np.arange(0.1, 0.8, 0.1),
        #     "affinity": ["precomputed", "euclidean"]
        # }

        # def custome_score(estimator, x_train):
        #     return silhouette_score(estimator.fit_predict(x_train), x_train)

        # search = GridSearchCV (
        #     AffinityPropagation(),
        #     param_distributions=param,
        #     scoring=custome_score,
        #     cv=4,
        #     n_iter=X_TIME,
        #     verbose=0)
        # search.fit(self.x_train)
        # affinity_model = AffinityPropagation(
        #     damping=search.best_params_["damping"],
        #     affinity=search.best_params_["affinity"]
        # )
        return AffinityPropagation().fit(self.x_train), "Add_best_later"

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
        # param = {
        #     "eps": np.arange(0.1, 0.7, 0.1),
        #     "min_samples": [1, 5, 10, 15],
        # }
        # search = RandomizedSearchCV(
        #     dbscan_model,
        #     param_distributions=param,
        #     n_iter=X_TIME,
        #     verbose=0
        # )
        # search.fit(self.x_train)
        return dbscan_model.fit(self.x_train), "Add_best_later"

    def AGC(self):
        agc = AgglomerativeClustering(
            n_clusters=5, affinity="euclidean", linkage="ward"
        )
        return agc.fit(self.x_train), "Add_best_later"

    def MS(self):
        """Add RandomizedSearchCV later"""
        return MeanShift().fit(self.x_train), "Add_best_later"

    def SPC(self):
        return SpectralClustering(n_clusters=5).fit(self.x_train), "Add_best_later"
