import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

import wandb


@dataclass(slots=True)
class Model(ABC):
    name: str

    @abstractmethod
    def predict(self, X: pd.DataFrame, target_name: str) -> pd.Series:
        ...

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        ...


@dataclass(slots=True)
class Validator:

    train: pd.DataFrame
    holdout: pd.DataFrame

    targets: List[str] = field(default_factory=list)

    project_name: str = 'justplay-pltv'

    logger: logging.Logger = logging.getLogger(__name__)

    def __post_init__(self):
        if not self.targets:
            self.targets = ['ltv_day1', 'ltv_day7', 'ltv_day30', 'ltv_day180']

    def get_metrics(self, model: Model, target: str) -> Dict[str, float]:
        train_pred = model.predict(self.train, target)
        holdout_pred = model.predict(self.holdout, target)

        return {
            'train/rmse': root_mean_squared_error(
                self.train[target],
                train_pred,
            ),
            'holdout/rmse': root_mean_squared_error(
                self.holdout[target],
                holdout_pred,
            ),
            'train/mae': mean_absolute_error(
                self.train[target],
                train_pred,
            ),
            'holdout/mae': mean_absolute_error(
                self.holdout[target],
                holdout_pred,
            ),
            'train/wape': self.wape(self.train[target], train_pred),
            'holdout/wape': self.wape(self.holdout[target], holdout_pred),
        }

    def get_all_metrics(self, model: Model) -> Dict[str, Dict[str, float]]:
        return {
            target: self.get_metrics(model, target)
            for target in self.targets
        }

    def log_metrics(self, model: Model) -> None:
        for target, metrics in self.get_all_metrics(model).items():
            self.logger.info(f'{model.name} {target} metrics: {metrics}')
            run = wandb.init(
                project=self.project_name,
                reinit=True,
                name=model.name,
                tags=[model.name, target],
                config=model.get_params(),
                notes=target,
                group=target,
            )
            run.log(metrics)
            run.finish(quiet=True)

    @staticmethod
    def wape(y_true: pd.Series, y_pred: pd.Series) -> float:
        return np.abs(y_true - y_pred).sum() / y_true.abs().sum()
