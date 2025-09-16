import time, bisect
import numpy as np
import pandas as pd
from scipy.stats import kruskal

class ScottKnottESD:
    # Default parameters, from the original repo
    def __init__(self, df: pd.DataFrame, alpha: float = 0.05, effect_thresh: float = 0.147):
        self.df = df.copy()
        self.alpha = alpha
        self.effect_thresh = effect_thresh
        self.agg_df = pd.DataFrame()
        self.ordered_seqs = []
        self.clusters = []

    def cliffs_delta(self, x:float, y:float) -> float:
        # Sorted x and y in ascending order
        x_sorted, y_sorted = sorted(x), sorted(y)
        m, n = len(x_sorted), len(y_sorted)
        # Count the number of pairs (xi, yj) where xi > yj and xi < yj
        num_xi_gt = num_xi_lt = 0
        for xi in x_sorted:
            # Obtain the numbers of y_i that is lower than xi by locating the insertion index of xi in y_sorted
            num_y_lt = bisect.bisect_left(y_sorted, xi)
            # Obtain the numbers of y_i that is greater than xi 
            num_y_gt = n - bisect.bisect_right(y_sorted, xi)
            num_xi_gt += num_y_lt
            num_xi_lt += num_y_gt
        return abs((num_xi_gt - num_xi_lt) / (m * n)) # Compute Cliff's Delta

    def aggregate_and_sorted(self, groupby: str, agg_col: str, agg_func: str = 'median') -> pd.DataFrame:
            # Preprocessing: Aggregate and sort the Performance column
            self.group_by = groupby
            self.agg_col = agg_col
            self.agg_func = agg_func
            if self.agg_func == 'median':
                self.agg_stat_col = f'{agg_col}_median'
                self.agg_df = self.df.groupby(self.group_by)[self.agg_col].median().reset_index().sort_values(by=agg_col, ascending = True)
                self.ordered_seqs = self.agg_df[self.group_by].tolist()
            elif self.agg_func == 'mean':
                self.agg_stat_col = f'{agg_col}_mean'
                self.agg_df = self.df.groupby(self.group_by)[self.agg_col].mean().reset_index().sort_values(by=agg_col, ascending = True)
                self.ordered_seqs = self.agg_df[self.group_by].tolist()
            else:
                raise ValueError(f"Unknown aggregation function: {agg_func}")

            self.agg_df.rename(columns = {self.agg_col: self.agg_stat_col}, inplace=True)

    def _best_cut(self, block_seqs: list) -> tuple:
        best_idx, best_H, best_p = None, -1.0, None
        for cut in range(1, len(block_seqs)):
            # Tried separating at the cut index
            L, R = block_seqs[:cut], block_seqs[cut:]
            # Extract their values
            Lv = self.df[self.df[self.group_by].isin(L)][self.agg_col]
            Rv = self.df[self.df[self.group_by].isin(R)][self.agg_col]
            # Kruskal-Wallis test
            try:
                H, p = kruskal(Lv, Rv)
                # Larger h indicates a better separation, so updated if we found the new best
                if H > best_H:
                    best_idx, best_H, best_p = cut, H, p
            except ValueError as e:
                # This means that the performance of L and R are identical, so skip this cut
                print(f"Testing cut at {cut}: {L} | {R}")
                print(f"Left values: {Lv.tolist()}")
                print(f"Right values: {Rv.tolist()}")
                print(f"Error in Kruskal test for cut {cut}: {e}")
                continue
        return best_idx, best_p

    def _accept_split(self, left_seqs: list, right_seqs: list, p_value: float) -> bool:
        # First Condition: p-value from statistical test must be below alpha
        if p_value is None or p_value >= self.alpha:
            return False
        # Second Condition: Effect size must be above the threshold
        for sL in left_seqs:
            # Performance Values from each sequence in the left block
            x = self.df[self.df[self.group_by] == sL][self.agg_col].to_numpy()
            for sR in right_seqs:
                # Performance Values from each sequence in the right block
                y = self.df[self.df[self.group_by] == sR][self.agg_col].to_numpy()
                # Check if there exists at least one sequence pairs (From left and right ) provides a large enough effect size
                # If yes, we accept the split
                if self.cliffs_delta(x, y) >= self.effect_thresh:
                    return True
        return False

    def _recurse(self, block_seqs: list):
        # Based case
        if len(block_seqs) == 1:
            self.clusters.append(block_seqs)
            return
        # Find the index to split with the best statistic test p-value
        idx, p_val = self._best_cut(block_seqs)
        left, right = block_seqs[:idx], block_seqs[idx:]
        # Check if we should accept the split from the statistical test
        if self._accept_split(left, right, p_val):
            self._recurse(left)
            self._recurse(right)
        else:
            self.clusters.append(block_seqs)

    def run(self, agg_func: str = "median") -> None:
        self._recurse(self.ordered_seqs)


