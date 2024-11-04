import logging
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm


@dataclass(slots=True)
class MockDBClient:
    df: pd.DataFrame

    def select(
        self,
        columns: List[str],
        where: Dict[str, Any],
        limit: int = None,
    ) -> pd.DataFrame:
        """Select data from the database"""
        if columns[0] == '*':
            columns = self.df.columns

        df = self.df.copy()
        for col, val in where.items():
            df = df.loc[df[col] == val]
        return df[columns].head(limit or len(df))


@dataclass(slots=True)
class StratifiedTTest:
    strat_columns: List[str]
    db: MockDBClient

    alpha: float = 0.05
    beta: float = 0.2

    metric_column: str = 'converted'
    group_column: str = 'group'
    random_state: int = 648

    logger: logging.Logger = logging.getLogger(__name__)

    rng: np.random.Generator = field(
        default=np.random.default_rng(random_state),
        init=False,
    )

    def get_pvalue(self, a: pd.DataFrame, b: pd.DataFrame) -> float:
        """Return p-value of the test"""
        a_strat = a.groupby(self.strat_columns).agg(
            count=(self.metric_column, 'count'),
            mean=(self.metric_column, 'mean'),
            std=(self.metric_column, 'std'),
        )
        a_strat['weight'] = a_strat['count'] / len(a)

        b_strat = b.groupby(self.strat_columns).agg(
            count=(self.metric_column, 'count'),
            mean=(self.metric_column, 'mean'),
            std=(self.metric_column, 'std'),
        )
        b_strat['weight'] = b_strat['count'] / len(b)

        # Align strata
        common_strata = a_strat.index.intersection(b_strat.index)
        a_strat = a_strat.loc[common_strata]
        b_strat = b_strat.loc[common_strata]

        # Calculate weighted means
        a_strat_mean = (a_strat['mean'] * a_strat['weight']).sum()
        b_strat_mean = (b_strat['mean'] * b_strat['weight']).sum()

        # Calculate variance of the weighted means
        a_strat_var = (
            (a_strat['std'] ** 2) * (a_strat['weight'] ** 2)
            / a_strat['count']
        ).sum()
        b_strat_var = (
            (b_strat['std'] ** 2) * (b_strat['weight'] ** 2)
            / b_strat['count']
        ).sum()

        # Compute t-statistic
        t_stat = (
             (a_strat_mean - b_strat_mean)
             / np.sqrt(a_strat_var + b_strat_var)
        )

        # Degrees of freedom
        df = min(len(a) - 1, len(b) - 1)

        # Compute two-tailed p-value
        return stats.t.sf(np.abs(t_stat), df) * 2

    def get_sample_size(
        self,
        effects: List[float],
        sample_interval: int = 5,
        bs_iter: int = 100,
        early_stopping_rounds: int = 50,
        max_reshuffle: int = 10_000,
    ) -> Dict[float, Tuple[int, Dict[str, np.ndarray]]]:
        """Estimate sample size for a list of effects"""
        df = self.db.select(
            self.strat_columns + [self.metric_column],
            {self.group_column: 'A'},
        )

        for i in range(max_reshuffle):
            ixs = self.rng.permutation(df.index)
            control = df.loc[ixs[:len(ixs) // 2]]
            test = df.loc[ixs[len(ixs) // 2:]]
            logging.debug(
                f'C: {control[self.metric_column].mean()},'
                f' T: {test[self.metric_column].mean()}',
            )
            if self.get_pvalue(control, test) > self.alpha:
                break
            else:
                logging.info(f'Reshuffling AA test. ({i})')
        control.set_index(self.strat_columns, inplace=True)
        test.set_index(self.strat_columns, inplace=True)

        sample_sizes = {}
        for e in tqdm(effects, desc='Iterating over effects'):
            for i in range(max_reshuffle):
                test = self._generate_synthetic_test(control, test, e)
                actual_effect = (
                    test[self.metric_column].mean()
                    / control[self.metric_column].mean()
                )
                if np.isclose(actual_effect, e, rtol=0.001):
                    logging.info(
                        f'Created synthetic test. '
                        f"Expected: {e}, Actual: {actual_effect}. "
                    )
                    break
                else:
                    logging.info(
                        'Reshuffling synthetic. '
                        f'Expected: {e}, Actual: {actual_effect}. '
                        f'({i})'
                    )
            sample_size, plot = self._calc_sample_size(
                control.reset_index(),
                test.reset_index(),
                sample_interval,
                bs_iter,
                early_stopping_rounds,
            )
            sample_sizes[e] = (sample_size, plot)
        return sample_sizes

    def _calc_sample_size(
        self,
        control: pd.DataFrame,
        test: pd.DataFrame,
        sample_interval: int = 5,
        bs_iter: int = 100,
        early_stopping_rounds: int = 50,
    ) -> Tuple[int, Dict[str, np.ndarray]]:
        """
        Return the minimal sample size to detect the effect.
        In case of no sample size found, return -1.
        """
        logging.debug(
            f'C: {control[self.metric_column].mean()},'
            f' T: {test[self.metric_column].mean()}',
        )
        max_size = min(len(control), len(test)) * early_stopping_rounds

        sample_size = -1
        pos_rounds = 0

        sizes = []
        powers = []
        it = tqdm(
            range(sample_interval, max_size, sample_interval),
            desc='Run simulation',
        )
        for i in it:
            pvals = []
            for _ in range(bs_iter):
                control_sample = self.rng.choice(
                    control.index,
                    size=i,
                    replace=True,
                )
                test_sample = self.rng.choice(
                    test.index,
                    size=i,
                    replace=True,
                )
                pval = self.get_pvalue(
                    control.loc[control_sample],
                    test.loc[test_sample],
                )
                pvals.append(pval)
            power = np.mean(np.array(pvals) < self.alpha)
            logging.debug(f'{i}. Power: {power}, pval: {np.mean(pvals)}')
            sizes.append(i)
            powers.append(power)

            if power >= 1 - self.beta:
                pos_rounds += 1
            else:
                pos_rounds = 0

            if pos_rounds >= early_stopping_rounds:
                sample_size = i - (early_stopping_rounds - 1) * sample_interval
                break


        return (
            sample_size,
            {'x': np.array(sizes), 'y': np.array(powers)},
        )


    def _generate_synthetic_test(
        self,
        control: pd.DataFrame,
        test: pd.DataFrame,
        e: float,
    ) -> pd.DataFrame:
        test = test.copy()
        for ixs, strata in test.groupby(test.index):
            metric = self._apply_effect(
                strata,
                control_level=control.loc[ixs, self.metric_column].mean(),
                effect=e,
            )
            test.loc[ixs, self.metric_column] = metric
        return test

    def _apply_effect(
        self,
        df: pd.DataFrame,
        control_level: float,
        effect: float,
    ) -> np.ndarray:
        return np.random.binomial(
            1,
            control_level * effect,
            size=len(df),
        )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    warnings.simplefilter(
        action='ignore',
        category=pd.errors.PerformanceWarning,
    )

    db = MockDBClient(
        df=pd.read_csv(
            "https://docs.google.com/spreadsheets/d/"
            "1U7WJihimU4nSjv9QJR7mdj2eUvSY_iHVq0FiC_TlnSs/"
            "gviz/tq?tqx=out:csv"
        ),
    )
    ttest = StratifiedTTest(
        strat_columns=['location', 'age_group'],
        db=db,
    )
    sample_size = ttest.get_sample_size(
        [1.1, 1.05, 1.01],
        sample_interval=100,
        bs_iter=100,
        early_stopping_rounds=10,
    )

    control = db.select(['*'], {'group': 'A'})
    test = db.select(['*'], {'group': 'B'})

    print('Sample sizes:', {e: pack[0] for e, pack in sample_size.items()})
    print('P-Value of the test presented: ', ttest.get_pvalue(control, test))
