"""gcForest (Deep Forest), cascade part only (no multi-grained scanning on tabular data).

Each level trains a RandomForest + ExtraTrees on [input features, the class-probability vectors of all previous
levels] (the vectors of the training rows are the forests' in-sample predictions). The cascade grows until the AUC of a
holdout stops improving (patience levels); the prediction is the mean P(y=1) of the best level's two forests."""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score


class GCForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=5, max_features=0.5, max_levels=5,
                 patience=1, seed=777, n_jobs=16):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_levels = max_levels
        self.patience = patience
        self.seed = seed
        self.n_jobs = n_jobs
        self.levels = []
        self.best_level = 0
        self.best_auc = 0.0

    def _make_forests(self):
        f1 = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
            n_jobs=self.n_jobs, random_state=self.seed)
        f2 = ExtraTreesClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
            n_jobs=self.n_jobs, random_state=self.seed + 1)
        return f1, f2

    def fit(self, X, y, X_val, y_val):
        """Grow the cascade; X grows with each level's class vectors."""
        self.levels = []
        best_auc = 0.0
        best_level = 0
        for level in range(self.max_levels):
            f1, f2 = self._make_forests()
            f1.fit(X, y)
            f2.fit(X, y)
            self.levels.append((f1, f2))
            p1 = f1.predict_proba(X_val)
            p2 = f2.predict_proba(X_val)
            auc = roc_auc_score(y_val, (p1[:, 1] + p2[:, 1]) / 2)
            if auc > best_auc:
                best_auc = auc
                best_level = level
            elif level - best_level >= self.patience:
                break
            # augment with this level's class vectors (full 2-dim vectors)
            X = np.hstack([X, f1.predict_proba(X), f2.predict_proba(X)])
            X_val = np.hstack([X_val, p1, p2])
        self.best_level = best_level
        self.best_auc = best_auc
        return self

    def predict_proba(self, X):
        p = None
        for level in range(self.best_level + 1):
            f1, f2 = self.levels[level]
            p1 = f1.predict_proba(X)
            p2 = f2.predict_proba(X)
            p = (p1[:, 1] + p2[:, 1]) / 2
            if level < self.best_level:
                X = np.hstack([X, p1, p2])
        return p
