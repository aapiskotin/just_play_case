from dataclasses import field
from typing import List

import numpy as np
import pandas as pd
from attr import dataclass


@dataclass(slots=True)
class ThompsonSampling:
    variants: List[str]

    random_seed: int = 648

    rng: np.random.Generator = field(
        default=np.random.default_rng(random_seed),
        init=False,
    )
    stats: pd.DataFrame = field(init=False)

    def __post_init__(self):
        self.stats = pd.DataFrame(
            {'imp': 0, 'click': 0},
            index=self.variants,
        )

    def get_one_best(self):
        a, b = self.get_a_b()
        return self.variants[np.argmax(self.rng.beta(a, b))]

    def get_a_b(self):
        a = self.stats['click'].clip(lower=1)
        b = (self.stats['imp'] - self.stats['click']).clip(lower=1)
        return a, b

    def add_imp(self, variant):
        self.stats.loc[variant, 'imp'] += 1

    def add_click(self, variant):
        self.stats.loc[variant, 'click'] += 1
