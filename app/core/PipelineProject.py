import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
)


class PrepareDataAndTrainingModels:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        target: str,
        is_only_numeric_cols: bool = True,
        random_state: int = 42,
        test_size: float = 0.3,
        **kwargs,
    ) -> None:

        self.dataframe = dataframe
        self.target = target
        self.is_only_num_cols = is_only_numeric_cols
        self.random_state = random_state
        self.test_size = test_size
        self.kwargs = kwargs

    def get_name_cat_and_num_cols(self):
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

    def splitting_data(self):

        X = self.dataframe.drop(self.target, axis=1)
        Y = self.dataframe[self.target]
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, random_state=self.random_state, test_size=self.test_size
        )

        return X_train, X_test, Y_train, Y_test

    def pre_process_data(self):

        if self.is_only_num_cols:
            X_train, X_test, Y_train, Y_test = self.balance_data()
            X_train = self.kwargs["scaler"].fit_transform(X_train)
            X_test = self.kwargs["scaler"].fit_transform(X_test)

            return X_train, X_test, Y_train, Y_test

    def balance_data(self):

        qt_classes = (
            self.dataframe[self.target]
            .value_counts(ascending=True)
            .reset_index(drop=True)
        )
        X_train, X_test, Y_train, Y_test = self.splitting_data()
        if len(qt_classes) == 2:
            print("Its Binary Classification!\n\n")
            print("#" * 50 + "\n\n")
            if qt_classes[0] / qt_classes[1] < 0.35:
                print(f"Using tecnique: {self.kwargs['balancer']}")
                X_train, Y_train = self.kwargs["balancer"].fit_resample(
                    X_train,
                    Y_train,
                )
                return X_train, X_test, Y_train, Y_test
        return X_train, X_test, Y_train, Y_test

    def avaliate_models(self):

        score_models = dict()
        models = self.kwargs["models"]
        X_train, X_test, Y_train, Y_test = self.pre_process_data()
        for _, model in enumerate(models):
            score_models[str(model)] = dict()
            predictor = model
            predictor.fit(X_train, Y_train)
            prediction = predictor.predict(X_test)
            score_models[str(model)]["accuracy"] = accuracy_score(Y_test, prediction)
            score_models[str(model)]["precision"] = precision_score(Y_test, prediction)
            score_models[str(model)]["recall"] = recall_score(Y_test, prediction)
            score_models[str(model)]["f1_score"] = f1_score(Y_test, prediction)

        return (
            pd.DataFrame(score_models)
            .T.reset_index()
            .rename(columns={"index": "model"})
        )

    def avaliate_models_cv(self):
        score_models = dict()
        models = self.kwargs["models"]
        X_train, X_test, Y_train, Y_test = self.pre_process_data()
        for _, model in enumerate(models):
            print(f"Model training: {str(model)}\n\n")
            score_models[str(model)] = dict()
            predictor = model
            predictor.fit(X_train, Y_train)
            prediction = predictor.predict(X_test)
            score_models[str(model)]["accuracy"] = accuracy_score(Y_test, prediction)
            score_models[str(model)]["precision"] = precision_score(Y_test, prediction)
            score_models[str(model)]["recall"] = recall_score(Y_test, prediction)
            score_models[str(model)]["f1_score"] = f1_score(Y_test, prediction)
            scores_ = cross_val_score(
                predictor, X_train, Y_train, cv=5, scoring=self.kwargs["score_metric"]
            )
            print(f"Metric: {self.kwargs['score_metric']} - {scores_.mean()}")
            print(f"#" * 50)
        return (
            pd.DataFrame(score_models)
            .T.reset_index()
            .rename(columns={"index": "model"})
        )
