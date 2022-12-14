import pandas as pd
import pickle
import re
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt


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

        self.save_path = "models/"
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
        self.models_predictions = None
        self.kwargs = kwargs

    @staticmethod
    def persist_model(
        model,
        save_path: str,
    ) -> None:
        with open(save_path, "wb") as file:
            pickle.dump(model, file)

    @staticmethod
    def load_model(load_path: str):
        with open(load_path, "rb") as file:
            model = pickle.load(file)
        return model

    def splitting_data(self, standart_scale=False, **kwargs) -> None:
        X = self.dataframe.drop(self.target, axis=1)
        if standart_scale == True:
            X = preprocessing.StandardScaler().fit(X).transform(X)

        Y = self.dataframe[self.target]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, random_state=self.random_state, test_size=self.test_size, **kwargs
        )

    def balance_data(self):
        qt_classes = (
            self.dataframe[self.target]
            .value_counts(ascending=True)
            .reset_index(drop=True)
        )
        if len(qt_classes) == 2:
            print("Its Binary Classification!\n\n")
            print("#" * 50 + "\n\n")

            if (self.kwargs["balancer"] != None) and (
                qt_classes[0] / qt_classes[1] < 0.35
            ):
                print(f"Using tecnique: {self.kwargs['balancer']}")
                self.X_train, self.Y_train = self.kwargs["balancer"].fit_resample(
                    self.X_train,
                    self.Y_train,
                )

    def pre_process_data(self):
        if not self.X_train:
            self.splitting_data()

        if self.is_only_num_cols:
            self.balance_data()
            self.X_train = self.kwargs["scaler"].fit_transform(self.X_train)
            self.X_test = self.kwargs["scaler"].fit_transform(self.X_test)

    def fit_models(
        self,
        cross_val=False,
        grid_search_cv=False,
        persist=False,
        param=None,
        cv=5,
        n_jobs=-1,
    ) -> pd.DataFrame:
        models = self.kwargs["models"]
        fitted_models = dict()
        for _, model in enumerate(models):
            if grid_search_cv == True:
                if param == None:
                    raise AttributeError(
                        f"If grid_search_cv is True, param cannot be None. param = {param}."
                    )

                predictor = GridSearchCV(
                    estimator=model, param_grid=param, cv=cv, n_jobs=n_jobs
                )
                predictor.fit(self.X_train, self.Y_train)

            elif cross_val == True:
                predictor = cross_validate(
                    estimator=model,
                    X=self.X_train,
                    y=self.Y_train,
                    cv=cv,
                    n_jobs=n_jobs,
                )

            else:
                predictor = model.fit(self.X_train, self.Y_train)

            if persist == True:
                self.persist_model(predictor, f"{self.save_path}+{str(model)}.pkl")
            fitted_models[model.__class__.__name__] = predictor

        self.fitted_models = fitted_models

    def predict(self) -> pd.DataFrame:
        models = self.kwargs["models"]
        predictions = dict()
        for _, model in enumerate(models):
            predictions[model.__class__.__name__] = model.predict(
                self.X_test, self.Y_test
            )

        self.models_predictions = predictions

    def compute_scores(self, **kwargs) -> pd.DataFrame:
        score_models = dict()
        models = self.kwargs["models"]
        metrics = self.kwargs["score_metric"]
        for _, model in enumerate(models):
            score_models[model.__class__.__name__] = dict()
            predictor = model
            prediction = predictor.predict(self.X_test)
            for _, metric in enumerate(metrics):
                name_metric = re.sub(r"(^<f\w*n|(at (.*)))", "", str(metric)).strip()
                score_models[model.__class__.__name__][name_metric] = metric(
                    self.Y_test, prediction
                )

        return (
            pd.DataFrame(score_models)
            .T.reset_index()
            .rename(columns={"index": "model"})
        )

    def plot_roc_curves(
        self, cross_val: bool = False, grid_search_cv: bool = False, param=None
    ) -> plt:

        result_table = pd.DataFrame(columns=["classifiers", "fpr", "tpr", "auc"])
        # self.fit_models(cross_val=cross_val, grid_search_cv=grid_search_cv, param=None)
        for key, model in self.fitted_models.items():
            yproba = model.predict(self.X_test)

            fpr, tpr, _ = roc_curve(self.Y_test, yproba)
            auc = roc_auc_score(self.Y_test, yproba)

            result_table = result_table.append(
                {
                    "classifiers": model.__class__.__name__,
                    "fpr": fpr,
                    "tpr": tpr,
                    "auc": auc,
                },
                ignore_index=True,
            )

        # Set name of the classifiers as index labels
        result_table.set_index("classifiers", inplace=True)

        fig = plt.figure(figsize=(8, 6))

        for i in result_table.index:
            plt.plot(
                result_table.loc[i]["fpr"],
                result_table.loc[i]["tpr"],
                label="{}, AUC={:.3f}".format(i, result_table.loc[i]["auc"]),
            )

        plt.plot([0, 1], [0, 1], color="orange", linestyle="--")

        plt.xticks(np.arange(0.0, 1.1, step=0.1))
        plt.xlabel("False Positive Rate", fontsize=15)

        plt.yticks(np.arange(0.0, 1.1, step=0.1))
        plt.ylabel("True Positive Rate", fontsize=15)

        plt.title("ROC Curve Analysis", fontweight="bold", fontsize=15)
        plt.legend(prop={"size": 13}, loc="lower right")

        plt.show()
