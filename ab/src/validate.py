from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm


@dataclass(slots=True)
class StatTest(ABC):
    metric_column: str = 'converted'

    @abstractmethod
    def __call__(self, a: pd.DataFrame, b: pd.DataFrame) -> float:
        """Return p-value of the test"""
        ...


@dataclass(slots=True)
class Validator:
    df: pd.DataFrame
    alpha: float = 0.05
    beta: float = 0.2
    metric_column: str = 'converted'
    groups_column: str = 'group'
    dt_column: str = 'created_date'
    random_state: int = 648

    rng: np.random.Generator = np.random.default_rng(random_state)

    def test_pvalue(
        self,
        test: StatTest,
        group: str = 'A',
        bs_size: int = 10_000,
    ) -> Tuple[bool, float, np.ndarray]:
        """
        Return the result of passing test,
        p-value of KS Uniformity test and the distribution of p-values
        """
        ixs = self.df.loc[self.df[self.groups_column] == group].index

        a = self.rng.choice(ixs, size=(bs_size, len(ixs)), replace=True)
        b = self.rng.choice(ixs, size=(bs_size, len(ixs)), replace=True)

        pval_dist = np.array(
            [
                test(self.df.loc[a[i, :]], self.df.loc[b[i, :]])
                for i in tqdm(range(bs_size))
            ],
        )

        ks_pval = stats.kstest(pval_dist, 'uniform').pvalue

        return ks_pval > self.alpha, ks_pval, pval_dist

    def test_mde(
        self,
        test: StatTest,
        sample_interval: int = 5,
        bs_size: int = 100,
        early_stopping_rounds: int = 50,
    ) -> Tuple[int, Dict[str, np.ndarray]]:
        """Return the minimal sample size to detect the effect"""
        a_full = self.df.loc[self.df[self.groups_column] == 'A'].index
        b_full = self.df.loc[self.df[self.groups_column] == 'B'].index
        max_size = min(len(a_full), len(b_full))

        i = -1
        pos_rounds = 0

        sizes = []
        powers = []
        for i in tqdm(range(sample_interval, max_size, sample_interval)):
            pvals = []
            for _ in range(bs_size):
                a_sample = self.rng.choice(a_full, size=i, replace=True)
                b_sample = self.rng.choice(b_full, size=i, replace=True)
                pval = test(self.df.loc[a_sample], self.df.loc[b_sample])
                pvals.append(pval)
            power = np.mean(np.array(pvals) < self.alpha)
            sizes.append(i)
            powers.append(power)

            if power >= 1 - self.beta:
                pos_rounds += 1
            else:
                pos_rounds = 0

            if pos_rounds >= early_stopping_rounds:
                break
        return (
            i - early_stopping_rounds * sample_interval,
            {'x': np.array(sizes), 'y': np.array(powers)},
        )
