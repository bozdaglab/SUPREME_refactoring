import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from torch_geometric.data import Data
from xgboost import XGBClassifier


logger = logging.getLogger(__name__)


class MLModels:
    def __init__(self, model, x_train, y_train):
        self.model = model
        self.x_train = x_train
        self.y_train = x_train

    def train_ml_model_factory(self):
        models = {
            "LR": self.MLP,
            "RF": self.RF,
            "XGB": self.XGB,
            "SVM": self.SVM,
        }
        try:
            return models.get(self.model)()
        except:
            pass

    def MLP():
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
        return MLPClassifier(
            solver="adam",
            activation="relu",
            early_stopping=True,
            hidden_layer_sizes=search.best_params_["hidden_layer_sizes"],
        )

    def XGBoost():
        params = {
            "reg_alpha": range(0, 6, 1),
            "reg_lambda": range(1, 5, 1),
            "learning_rate": [0, 0.001, 0.01, 1],
        }
        fit_params = {
            "early_stopping_rounds": 10,
            "eval_metric": "mlogloss",
            "eval_set": [(X_train, y_train)],
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

        return XGBClassifier(
            use_label_encoder=False,
            objective="multi:softprob",
            eval_metric="mlogloss",
            verbosity=0,
            n_estimators=1000,
            fit_params=fit_params,
            reg_alpha=search.best_params_["reg_alpha"],
            reg_lambda=search.best_params_["reg_lambda"],
            learning_rate=search.best_params_["learning_rate"],
        )

    def RF():
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
        return RandomForestClassifier(n_estimators=search.best_params_["n_estimators"])

    def SVM():
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
        model = SVC(C=search.best_params_["C"], gamma=search.best_params_["gamma"])
