import logging
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from settings import X_TIME
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class MLModels:
    def __init__(self, model, x_train, y_train: Optional[pd.DataFrame] = None):
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
            "KM": self.KM,
        }
        try:
            return models.get(self.model)()
        except:
            pass

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
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        params = {
            "n_estimators": [int(x) for x in np.linspace(start=200, stop=2000, num=100)]
        }
        search = RandomizedSearchCV(
            estimator=RandomForestClassifier(),
            return_train_score=True,
            scoring="f1_macro",
            param_distributions=params,
            cv=4,
            n_iter=X_TIME,
            verbose=0,
        )
        search.fit(self.x_train, self.y_train)
        return (
            RandomForestClassifier(n_estimators=search.best_params_["n_estimators"]),
            search,
        )

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

    def KM(self):
        params = {
            "n_clusters": [5],
            "init": ["k-means++", "random"],
            "n_init": [10, 20, 30],
        }

        search = RandomizedSearchCV(
            KMeans(), param_distributions=params, cv=4, n_iter=X_TIME, verbose=0,
        )

        search.fit(self.x_train)

        best_n_clusters = search.best_params_["n_clusters"]
        best_init = search.best_params_["init"]
        best_n_init = search.best_params_["n_init"]

        kmeans_model = KMeans(
            n_clusters=best_n_clusters, init=best_init, n_init=best_n_init
        )

        return kmeans_model, search
