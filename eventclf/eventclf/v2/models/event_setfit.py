from typing import Any, Callable, Dict, List, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from setfit import SetFitModel, Trainer, sample_dataset


class EventSetfit:
    def __init__(
        self,
        model: str,
        device: Union[str, torch.device] = "cpu",
        load_model: bool = False,
    ):
        """Inititalises event setfit model

        Parameters
        ----------
        model : str
            sentence tranformer to use as base for setfit can be local path for saved model
        device : Union[str, torch.device], optional
            device to run model on, by default "cpu"
        """
        self._model_id = model
        if load_model:
            self._model = SetFitModel.from_pretrained(model)
            self._model.to(device)

    def fit(
        self,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        label_column: str,
        text_column: str,
        training_sample_size: int,
        hp_space: Callable,
        n_trials: int = 10,
    ):
        """Performs hyperparameter search and finetuning of setfit model

        Parameters
        ----------
        train_df : pd.DataFrame
            dataframe of data to train on
        validation_df : pd.DataFrame
            dataframe of data to use as validation
        label_column : str
           name of column containing data label
        text_column : str
            name of column containing text input
        training_sample_size : int
            number of samples to train on
        hp_space : Callable
            Optuna function defining hyperparameter space
        """

        def model_init(params: Dict[str, Any]) -> SetFitModel:
            params = params or {}
            max_iter = params.get("max_iter", 100)
            solver = params.get("solver", "liblinear")
            params = {
                "head_params": {
                    "max_iter": max_iter,
                    "solver": solver,
                }
            }
            return SetFitModel.from_pretrained(self._model_id, **params)

        train_dataset = Dataset.from_pandas(train_df)
        train_dataset = sample_dataset(
            train_dataset, label_column=label_column, num_samples=training_sample_size
        )
        test_dataset = Dataset.from_pandas(validation_df)
        trainer = Trainer(
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            model_init=model_init,
            column_mapping={
                text_column: "text",
                label_column: "label",
            },
        )

        best_run = trainer.hyperparameter_search(
            direction="minimize", hp_space=hp_space, n_trials=n_trials
        )
        self._params = best_run.hyperparameters
        trainer.apply_hyperparameters(self._params, final_model=True)
        trainer.train()
        self._model = trainer.model

    def predict(self, text: List[str]) -> np.array:
        """Uses trained model to predict label of outut

        Parameters
        ----------
        text : List[str]
            list of inputs to classify

        Returns
        -------
        np.array
            binary array of classification
        """
        return np.array(self._model.predict(text))

    def save(self, outpath: str):
        """Saves local model

        Parameters
        ----------
        outpath : str
            path to output
        """
        self._model.save_pretrained(outpath)
