import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
    roc_auc_score,
)


class PrepareDataAndTrainingModels:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        target: str,
        is_only_numeric_cols: bool = True,
        random_state: int = 42,
        test_size: float = 0.3,
        balance: bool = False,
        **kwargs,
    ) -> None:

        self.dataframe = dataframe
        self.target = target
        self.is_only_num_cols = is_only_numeric_cols
        self.random_state = random_state
        self.test_size = test_size
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.fitted_models = None
        self.balance = balance
        self.kwargs = kwargs

    def get_name_cat_and_num_cols(self) -> None:
        """
        This method get names of columns categorical and numerical columns.
        """
        if self.is_only_num_cols:
            return (
                type(self.dataframe[self.dataframe.columns].columns),
                self.dataframe[self.dataframe.columns]
                .drop(self.target, axis=1)
                .columns,
            )

        possible_cat_cols = list()
        possible_cat_cols.append(self.target)
        for i, type_ in enumerate(self.dataframe.dtypes):
            if type_ == object:
                possible_cat_cols.append(self.dataframe.iloc[:, i].name)
            elif type_ == pd.CategoricalDtype.name:
                possible_cat_cols.append(self.dataframe.iloc[:, i].name)

        return type(
            (
                self.dataframe[possible_cat_cols].columns,
                self.dataframe.drop(possible_cat_cols, axis=1).columns,
            )
        ), (
            self.dataframe[possible_cat_cols].columns,
            self.dataframe.drop(possible_cat_cols, axis=1).columns,
        )

    def splitting_data(self, **kwargs) -> None:
        X = self.dataframe.drop(self.target, axis=1)
        Y = self.dataframe[self.target]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, random_state=self.random_state, test_size=self.test_size, **kwargs
        )

    def balance_data(self):

        print("#" * 50 + "\n\n")
        if self.balance == True:
            print(f"Using tecnique: {self.kwargs['balancer']}")
            self.X_train, self.Y_train = self.kwargs["balancer"].fit_resample(
                self.X_train,
                self.Y_train,
            )
        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def pre_process_data(self):
        if self.is_only_num_cols:
            self.X_train, self.X_test, self.Y_train, self.Y_test = self.balance_data()
            self.X_train = self.kwargs["scaler"].fit_transform(self.X_train)
            self.X_test = self.kwargs["scaler"].fit_transform(self.X_test)

    def cross_validation(self, predictor, **kwargs):
        return cross_val_score(predictor, self.X_train, self.Y_train, **kwargs)

    def fit_models(self, grid_search_cv=False, param=None, cv=5, n_jobs=-1) -> None:
        models = self.kwargs["models"]
        fitted_models = defaultdict(list)
        for _, model in enumerate(models):
            predictor = model
            if grid_search_cv == True:
                if param == None:
                    raise AttributeError(
                        f"If grid_search_cv is True, param cannot be None. param = {param}."
                    )
                predictor = GridSearchCV(predictor, param, cv, n_jobs)
                predictor.fit(self.X_train, self.Y_train)
                fitted_models["Model"].append(predictor.best_estimator_)

            else:
                predictor.fit(self.X_train, self.Y_train)
                fitted_models["Model"].append(predictor)

        self.fitted_models = pd.DataFrame(fitted_models)

    def fit_models_cv(self) -> None:
        models = self.kwargs["models"]
        fitted_models = dict()

        for _, model in enumerate(models):
            predictor = model
            scores_ = self.cross_validation(
                predictor, cv=5, scoring=self.kwargs["score_metric"]
            )
            print(f"Metric: {self.kwargs['score_metric']} - {scores_.mean()}")

        self.fitted_models = (
            pd.DataFrame(fitted_models)
            .T.reset_index()
            .rename(columns={"index": "model"})
        )

    def predict(self) -> pd.DataFrame:
        models = self.kwargs["models"]
        predictions = dict()
        for _, model in enumerate(models):
            predictor = model
            prediction = predictor.predict(self.X_test, self.Y_test)
            predictions[str(model)] = prediction

        return (
            pd.DataFrame(prediction).T.reset_index().rename(columns={"index": "model"})
        )

    def compute_scores(self, **kwargs) -> pd.DataFrame:
        score_models = dict()
        models = self.kwargs["models"]
        for _, model in enumerate(models):
            score_models[str(model)] = dict()
            predictor = model
            prediction = predictor.predict(self.X_test)
            score_models[str(model)]["accuracy"] = accuracy_score(
                self.Y_test, prediction
            )
            score_models[str(model)]["precision"] = precision_score(
                self.Y_test, prediction
            )
            score_models[str(model)]["recall"] = recall_score(self.Y_test, prediction)
            score_models[str(model)]["f1_score"] = f1_score(self.Y_test, prediction)
            score_models[str(model)]["roc_auc_score"] = roc_auc_score(
                self.Y_test, prediction
            )

        return (
            pd.DataFrame(score_models).reset_index().rename(columns={"index": "model"})
        )
