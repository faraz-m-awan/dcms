import os
import pickle
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


class LocationDataModel:
    def __init__(
        self,
        model: Literal["linear_model", "SVM", "random_forest", "xgb_model"],
        load_path: Optional[str] = None,
        scaler_path_x: Optional[str] = None,
        scaler_path_y: Optional[str] = None,
    ):
        """Initialises LocationDataModel based on models tested in initial analysis.
        Predicts attendence at events or sites based on Huq Data

        Parameters
        ----------
        model : Literal[linear_model,SVM,random_forest,xgb_model]
            string of model to be loaded
        load_path : Optional[str]
            path to pretrained model
        scaler_path_x : Optional[str]
            path to scaler for x variables, only when loading pretrained SVM model
        scaler_path_y : Optional[str]
            path to scaler for y variables, only when loading pretrained SVM model

        Raises
        ------
        ValueError
            raises error if model is not in predefined model list
        ValueError
            raises error if svm is loaded without scaler paths
        """

        location_data_base_models = {
            "linear_model": LinearRegression(),
            "SVM": SVR(kernel="linear", C=1.5, epsilon=0.001),
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "xgb_model": XGBRegressor(
                n_estimators=100,  # Number of trees
                learning_rate=0.1,  # Step size
                max_depth=3,  # Maximum depth of a tree
                objective="reg:squarederror",  # Objective for regression
                random_state=42,
            ),
        }

        if model not in location_data_base_models.keys():
            raise ValueError(
                f"model is not in model types {location_data_base_models.keys()}"
            )
        if (
            (model == "SVM")
            and load_path
            and any(not path for path in [scaler_path_x, scaler_path_y])
        ):
            raise ValueError(
                "trying to load SVM without loading associated trained scalers"
            )
        self.model_type = model
        if load_path:
            if self.model_type == "xgb_model":
                self.model = XGBRegressor()
                self.model.load_model(load_path)
            else:
                with open(load_path, "rb") as f:
                    self.model = pickle.load(f)
            if self.model_type == "SVM":
                with open(scaler_path_x, "rb") as f:
                    self.scaler_x = pickle.load(f)
                with open(scaler_path_y, "rb") as f:
                    self.scaler_y = pickle.load(f)
        else:
            self.model = location_data_base_models[model]
            if self.model_type == "SVM":
                self.scaler_x = StandardScaler()
                self.scaler_y = StandardScaler()

    def fit(self, X: Union[pd.Series, np.array], y: Union[pd.Series, np.array]):
        """Fits location model

        Parameters
        ----------
        X : Union[pd.series,np.array]
            X variables to train model
        y : Union[pd.series,np.array]
            values to predict
        """
        if self.model_type == "SVM":
            X_train = self.scaler_x.fit_transform(X.values)
            y_train = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()
        else:
            X_train = X
            y_train = y
        self.model.fit(X_train, y_train)

    def predict(self, X: Union[pd.Series, np.array]) -> np.array:
        """Predicts attendence at events

        Parameters
        ----------
        X : Union[pd.series,np.array]
            X values to base prediction on

        Returns
        -------
        np.array
            predicted attendences
        """
        if self.model_type == "SVM":
            X = self.scaler_x.transform(X)
        y_pred = self.model.predict(X)
        if self.model_type == "SVM":
            y_pred = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        return y_pred

    def save(self, model_path: str):
        """Saves either a pkl or json (xgb_models) at
        `model_path/model_name.pkl` or `model_path/model_name.json`

        Parameters
        ----------
        model_path : str
            path to save model output to
        """
        if self.model_type == "xgb_model":
            model_save = os.path.join(model_path, f"{self.model_type}.json")
            self.model.save_model(model_save)
        else:
            model_save = os.path.join(model_path, f"{self.model_type}.pkl")
            with open(model_save, "wb") as f:
                pickle.dump(self.model, f)
            if self.model_type == "SVM":
                scaler_x_path = os.path.join(model_path, "scaler_x.pkl")
                scaler_y_path = os.path.join(model_path, "scaler_y.pkl")
                with open(scaler_x_path, "wb") as f:
                    pickle.dump(self.scaler_x, f)
                with open(scaler_y_path, "wb") as f:
                    pickle.dump(self.scaler_y, f)
        print(f"Saving model at {model_save}")
