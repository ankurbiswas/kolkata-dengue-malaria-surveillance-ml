"""
fastboot.py -- fast and exact bootstrap helpers, shared by analyze_results.py and extra_analysis.py

A bootstrap resample of n cases is described by w = how many times each case is drawn (np.bincount of the drawn indices).
AUC and the confusion counts then follow from weighted sums over a ranking that is computed once, so no re-sorting is needed.
The numbers are identical to resampling the cases and calling scikit-learn (checked to 1e-16) and about 10 times faster.
"""
import numpy as np


def cm_from_counts(tp, fp, fn, tn):
    """Accuracy, Macro-F1, recall, precision, specificity from (possibly weighted) confusion counts. Dengue is the positive class."""
    f1d = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    f1m = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else np.nan
    return {"accuracy": (tp + tn) / (tp + fp + fn + tn), "macro_f1": (f1d + f1m) / 2, "recall": tp / (tp + fn) if (tp + fn) else np.nan,
            "precision": tp / (tp + fp) if (tp + fp) else np.nan, "specificity": tn / (tn + fp) if (tn + fp) else np.nan}


class Scorer:
    """One (outcome y, score p) pair; the ranking of p is computed once."""

    def __init__(self, y, p):
        self.y = np.asarray(y, dtype=float)
        self.p = np.asarray(p, dtype=float)
        _, self.inv = np.unique(self.p, return_inverse=True)   # ties share one group
        self.G = int(self.inv.max()) + 1

    def auc(self, w):
        wp = np.bincount(self.inv, weights=w * self.y, minlength=self.G)
        wn = np.bincount(self.inv, weights=w * (1 - self.y), minlength=self.G)
        cum = np.cumsum(wn) - wn
        return float((wp * (cum + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))

    def counts(self, t, w):
        """(tp, fp, fn, tn) at threshold t (prediction = dengue when p >= t)."""
        pred = self.p >= t
        return (float((w * self.y * pred).sum()), float((w * (1 - self.y) * pred).sum()),
                float((w * self.y * ~pred).sum()), float((w * (1 - self.y) * ~pred).sum()))

    def brier(self, w):
        return float((w * (self.p - self.y) ** 2).sum() / w.sum())


def draw_weights(rng, n):
    """Weights of one bootstrap resample of n cases (consumes the same random numbers as rng.integers(0, n, n))."""
    return np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
