import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn import preprocessing


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
            fitted_models[str(model)] = predictor

        self.fitted_models = fitted_models

    def predict(self) -> pd.DataFrame:
        models = self.kwargs["models"]
        predictions = dict()
        for _, model in enumerate(models):
            predictions[str(model)] = model.predict(self.X_test, self.Y_test)

        self.models_predictions = predictions

    def compute_scores(self, **kwargs) -> pd.DataFrame:
        score_models = dict()
        models = self.kwargs["models"]
        metrics = self.kwargs["score_metric"]
        for _, model in enumerate(models):
            score_models[str(model)] = dict()
            predictor = model
            prediction = predictor.predict(self.X_test)
            for _, metric in enumerate(metrics):
                score_models[str(model)][metric] = metric(self.Y_test, prediction)

        return (
            pd.DataFrame(score_models)
            .T.reset_index()
            .rename(columns={"index": "model"})
        )
