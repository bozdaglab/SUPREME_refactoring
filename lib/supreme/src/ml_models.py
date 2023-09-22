import logging
from sklearn.cluster import KMeans
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from settings import X_TIME

logger = logging.getLogger(__name__)


class MLModels:
    def __init__(self, model, x_train, y_train):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train

    def train_ml_model_factory(self):
        models = {
            "MLP": self.MLP,
            "RF": self.RF,
            "XGB": self.XGB,
            "SVM": self.SVM,
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
        
        kmeans_model = KMeans(n_clusters=best_n_clusters, init=best_init, n_init=best_n_init)
    
        return kmeans_model, search