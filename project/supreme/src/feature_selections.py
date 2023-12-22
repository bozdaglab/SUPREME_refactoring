import logging
from collections import defaultdict
from typing import List, Optional, Tuple

# import lime
import numpy as np
import pandas as pd
from helper import (
    drop_rows,
    features_ratio,
    load_models1,
    load_models2,
    lower_upper_bound,
    search_dictionary,
)
from learning_types import FeatureSelectionType
from ml_models import MLModels
from settings import MODELS_B, NUMBER_FEATURES, SELECTION_METHOD, X_ITER
from sklearn.feature_selection import SelectKBest, mutual_info_classif, r_regression
from sklearn.linear_model import Lasso
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger()


class FeatureALgo:
    def __init__(self) -> None:
        self.selector = {
            "pearson": self.select_pearson,
            "lasso": self.lasso,
            "mutual_information": self.mutual_information,
        }

    def select_pearson(self, X: pd.DataFrame, y: pd.DataFrame) -> List[str]:
        # replace r_regression with spearmanr (import from scipy)
        pearson_features = defaultdict()
        for k_val in NUMBER_FEATURES:
            pearson_features[k_val] = (
                SelectKBest(r_regression, k=k_val).fit(X, y).get_feature_names_out()
            )
        return search_dictionary(
            methods_features=pearson_features, thr=features_ratio(len(NUMBER_FEATURES))
        )

    def lasso(self, X: pd.DataFrame, y: pd.DataFrame, coef=0.00001) -> List[str]:
        lasso_features = defaultdict()
        lasso = Lasso()
        prev_param_grid = {"alpha": [0.01, 0.001, 0.1, 1.0]}
        for iter_number in range(1, X_ITER):
            logger.info(f"number of iteration for lasso: {iter_number}")
            scorer = make_scorer(mean_squared_error)
            lasso_gridsearch = GridSearchCV(
                lasso, prev_param_grid, scoring=scorer, cv=5
            )
            lasso_gridsearch.fit(X, y)
            best_alpha = lasso_gridsearch.best_estimator_.alpha
            prev_param_grid = {"alpha": lower_upper_bound(best_alpha)}
            features = X.columns[lasso_gridsearch.best_estimator_.coef_ > coef]
            if len(features):
                lasso_features[f"{iter_number}"] = features
        if lasso_features:
            logger.info("Done with Lasso")
            return search_dictionary(
                methods_features=lasso_features, thr=features_ratio(X_ITER)
            )
        return None

    # def select_lime(self, X: pd.DataFrame, y: pd.DataFrame, mlmodel: str) -> List[str]:
    #     X_featurenames = X.columns
    #     new_label = y.apply(lambda x: 2 if x > 0.5 else 1 if x == 0.5 else x)
    #     model = MLModels(model=mlmodel, X=X, y=y).train_classifier()

    #     explainer = lime.lime_tabular.LimeTabularExplainer(
    #         np.array(X),
    #         feature_names=X_featurenames,
    #         class_names=new_label.unique(),
    #         # categorical_features=,
    #         verbose=True,
    #         mode="classification",
    #     )
    #     exp = explainer.explain_instance(X.loc[0], model.predict_proba, top_labels=3)
    #     exp.as_map()

    def select_features_models1(
        self,
        features: pd.DataFrame,
        label: pd.DataFrame,
        feature_selection_type: str,
        ml_model_train,
    ) -> List[str]:
        rfe_features = defaultdict()
        for feature_number in NUMBER_FEATURES:
            logger.info(f"Select {feature_number} features using  RF")

            feature_selector = load_models1(
                feature_selection_type=feature_selection_type,
                mlmodel=ml_model_train,
                feature_number=feature_number,
            )
            if feature_selector.estimator:
                fitted_features = feature_selector.fit(features, label)
                try:
                    final_features = fitted_features.get_feature_names_out()
                except:
                    final_features = fitted_features.k_feature_names_

                rfe_features[f"RF_{feature_number}"] = final_features
        if rfe_features:
            logger.info("Done with models and features set")
            return search_dictionary(
                methods_features=rfe_features, thr=features_ratio(len(NUMBER_FEATURES))
            )
        return None

    def selec_feature_models2(
        self,
        features: pd.DataFrame,
        label: pd.DataFrame,
        feature_selection_type: str,
        ml_model_train,
    ) -> List[str]:
        single_feature = defaultdict()
        featute_selector = load_models2(
            feature_selection_type=feature_selection_type,
            mlmodel=ml_model_train,
        )
        try:
            fitted_features = featute_selector.fit(features, label)
        except:
            fitted_features = featute_selector.fit(np.array(features), np.array(label))
        try:
            final_features = fitted_features.get_feature_names_out()
        except:
            final_features = features.columns[fitted_features.support_]
        single_feature["model"] = final_features

        if single_feature:
            logger.info("Done with models and features set")
            return search_dictionary(
                methods_features=single_feature,
                thr=features_ratio(len(SELECTION_METHOD)),
            )
        return None

    def mutual_information(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        mi_result = mutual_info_classif(X, y, random_state=42)
        features_list = X.columns.to_list()
        last_featutres = [f for f, v in zip(features_list, mi_result) if v > 0]
        if len(last_featutres) == len(X.columns):
            for i in X:
                if i not in last_featutres:
                    X = X.drop(i, axis=1)
        return X


def select_features(
    application_train: pd.DataFrame,
    labels: pd.DataFrame,
    feature_type: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ml_model_train = MLModels(
        model=MODELS_B[0], x_train=application_train, y_train=labels
    ).train_classifier()
    methods_features = defaultdict(lambda: defaultdict(int))
    all_methods = FeatureALgo()
    if isinstance(feature_type, list):
        for method in feature_type:
            select_features_ = apply_features_selections(
                method=method,
                all_methods=all_methods,
                application_train=application_train,
                labels=labels,
                ml_model_train=ml_model_train,
            )
            try:
                if any(select_features_):
                    for feature in select_features_:
                        methods_features[f"{method}"][feature] += 1
            except TypeError:
                pass
        final_features = search_dictionary(
            methods_features, thr=features_ratio(len(methods_features))
        )
    else:
        final_features = apply_features_selections(
            method=feature_type,
            all_methods=all_methods,
            application_train=application_train,
            labels=labels,
            ml_model_train=ml_model_train,
        )
    if final_features:
        return drop_rows(application_train, final_features), final_features
    # application_train = all_methods.mutual_information(application_train, y)
    return [], []


def apply_features_selections(
    method: str,
    all_methods: FeatureALgo,
    application_train: pd.DataFrame,
    labels: pd.DataFrame,
    ml_model_train: MLModels,
):
    if method in [
        FeatureSelectionType.BorutaPy.name,
        FeatureSelectionType.SelectBySingleFeaturePerformance.name,
        FeatureSelectionType.SelectByShuffling.name,
        FeatureSelectionType.GeneticSelectionCV.name,
    ]:
        return all_methods.selec_feature_models2(
            features=application_train,
            label=labels,
            feature_selection_type=method,
            ml_model_train=ml_model_train,
        )

    elif method in [
        FeatureSelectionType.RFE.name,
        FeatureSelectionType.SelectFromModel.name,
        FeatureSelectionType.SequentialFeatureSelector.name,
    ]:
        return all_methods.select_features_models1(
            features=application_train,
            label=labels,
            feature_selection_type=method,
            ml_model_train=ml_model_train,
        )
    else:
        selector_method = all_methods.selector.get(method)
        return selector_method(application_train, labels)
