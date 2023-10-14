import numpy as np
import pandas as pd
from feature_engine.selection import SelectByShuffling, SelectBySingleFeaturePerformance
from lightgbm import LGBMClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    chi2,
    mutual_info_classif,
    r_regression,
)
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from boruta import BorutaPy
from sklearn.metrics import accuracy_score, make_scorer

from sklearn.linear_model import LogisticRegression, Lasso

class FeatureALgo:
    def __init__(self) -> None:
        self.selector = {
            "pearson": self._select_pearson,
            "rfe": self._select_rff,
            "sfmrf": self._selec_sfmrf,
            "sfmlgb": self._select_sfmlgb,
            "sequemtial": self._selec_sequemtial,
            "chi": self._select_chi,
            "single_feture": self._selec_by_single_feature,
            "shuffle_feature": self._selec_by_shafffling_feature,
            "lasso": self._lasso,
            "boruta": self._select_boruta,
        }

    def select_pearson(self, X, y):
        return SelectKBest(r_regression, k=2).fit(X, y).get_feature_names_out()

    def lasso(self, X: pd.DataFrame, y: pd.DataFrame, coef=0.00001) -> List[str]:
        lasso = Lasso()
        param_grid = {"alpha": [0.01]}
        scorer = make_scorer(accuracy_score)
        lasso_gridsearch = GridSearchCV(lasso, param_grid, scoring=scorer, cv=5)
        lasso_gridsearch.fit(X, y)
        return X.columns[lasso_gridsearch.best_estimator_.coef_ > coef]
    
    def select_rff(self, X, y):
        rfe_selector = RFE(
            estimator=LogisticRegression(),
            n_features_to_select=2,
            step=250,
            verbose=5,
        )
        return rfe_selector.fit(X, y).get_feature_names_out()

    def selec_sfmrf(self, X, y):
        embeded_rf_selector = SelectFromModel(
            RandomForestClassifier(n_estimators=200), threshold="1.25*median"
        )
        return embeded_rf_selector.fit(X, y).get_feature_names_out()

    def select_sfmlgb(self, X, y):
        lgbc = LGBMClassifier(
            n_estimators=450,
            learning_rate=0.05,
            num_leaves=32,
            colsample_bytree=0.2,
            reg_alpha=3,
            reg_lambda=1,
            min_split_gain=0.01,
            min_child_weight=40,
        )
        embeded_lgb_selector = SelectFromModel(lgbc, threshold="1.25*median")
        return embeded_lgb_selector.fit(X, y).get_feature_names_out()

    def select_chi(self, X, y):
        chi_selector = SelectKBest(chi2, k=2)
        return chi_selector.fit(X, y).get_feature_names_out()

    def selec_by_single_feature(self, X, y):
        selected_by_single = SelectBySingleFeaturePerformance(
            estimator=RandomForestClassifier(
                n_estimators=10, max_depth=2, random_state=1
            ),
            scoring="roc_auc",
            cv=3,
            threshold=0.6,
        )

        return selected_by_single.fit(X, y).get_feature_names_out()

    def selec_by_shafffling_feature(self, X, y):
        feature_shuffle = SelectByShuffling(
            estimator=RandomForestClassifier(
                n_estimators=10, max_depth=2, random_state=1
            ),
            scoring="roc_auc",
            cv=3,
            threshold=0.06,
        )

        return feature_shuffle.fit(X, y).get_feature_names_out()

    def selec_sequemtial(self, X, y):
        sfsmodel = sfs(
            RandomForestRegressor(),
            k_features=2,
            forward=True,
            verbose=2,
            cv=5,
            n_jobs=2,
            scoring="r2",
        )
        return sfsmodel.fit(X, y)

    def select_boruta(self, X: pd.DataFrame, y: pd.DataFrame) -> List[str]:
        model = xgb.XGBClassifier()
        features = BorutaPy(
            estimator=model,
            n_estimators="auto",
            verbose=2,
            random_state=42,
            max_iter=40,
        )
        features.fit(np.array(X), np.array(y))
        selected_features = features.support_
        return [
            X.columns[i] for i in range(len(selected_features)) if selected_features[i]
        ]
    
    def MI(application_train, y):
        mi_result = mutual_info_classif(
            application_train, y, random_state=42
        )
        features_list = application_train.columns.to_list()
        last_featutres = [f for f, v in zip(features_list, mi_result) if v > 0]
        for i in application_train:
            if i not in last_featutres:
                application_train = application_train.drop(i, axis=1)
        return application_train



def drop_rows(application_train: pd.DataFrame, gh) -> pd.DataFrame:
    """
    drop the rows and columns that have only 0
    """
    for i in application_train:
        if i not in gh:
            application_train = application_train.drop(i, axis=1)
    a_series = (application_train == 0).all(axis=1)
    for index_to_remove in application_train.loc[a_series].index:
        application_train = application_train.drop(index_to_remove)
    return application_train.reset_index(drop=True)

def select_features(application_train, numb_thr=2):
    y = application_train["Survived"]
    X = application_train.drop(["Survived"], axis=1)
    feature_selection = FeatureALgo()
    f = {"features": X.columns}
    for selector_name, selector_method in feature_selection.selector.items():
        select_features = selector_method(X, y)
        f[selector_name] = [feature in select_features for feature in X.columns]
    feature_selection_df = pd.DataFrame(f)
    feature_selection_df["Total"] = np.sum(feature_selection_df, axis=1)

    gh = list(
        feature_selection_df.loc[
            feature_selection_df["Total"] > numb_thr, "features"
        ].values
    )
    application_train = drop_rows(X, gh)
    application_train = feature_selection.MI(application_train, y)
    application_train[y.name] = y
    return  application_train


