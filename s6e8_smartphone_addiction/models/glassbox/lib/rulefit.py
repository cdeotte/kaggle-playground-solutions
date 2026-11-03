"""RuleFit: rule conjunctions + a sparse linear model.

1. Train a RandomForest + ExtraTrees (max_features 0.5) on the features.
2. Every root-to-leaf path of every tree is a rule (a conjunction of split conditions); duplicates are merged.
3. Binary firing matrix: 1 if a row satisfies the rule.
4. Elastic-net logistic regression (SGD) on [features, rule firings], standardized on the training rows."""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


class RuleFit:
    def __init__(self, n_estimators=100, max_depth=5, min_samples_leaf=10, alpha=1e-4, l1_ratio=0.9, seed=777,
                 n_jobs=16):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.seed = seed
        self.n_jobs = n_jobs
        self.rules = []
        self.scaler = None
        self.lin = None

    @staticmethod
    def _extract_rules(forests):
        rules = set()
        for forest in forests:
            for est in forest.estimators_:
                t = est.tree_

                def walk(node, conds):
                    if t.children_left[node] == -1:  # leaf
                        if conds:
                            rules.add(tuple(conds))
                        return
                    f = t.feature[node]
                    thr = t.threshold[node]
                    walk(t.children_left[node], conds + [(int(f), float(thr), True)])
                    walk(t.children_right[node], conds + [(int(f), float(thr), False)])

                walk(0, [])
        return list(rules)

    @staticmethod
    def _fire(X, rules):
        M = np.zeros((X.shape[0], len(rules)), dtype=np.float32)
        for j, rule in enumerate(rules):
            fire = np.ones(X.shape[0], dtype=bool)
            for feat, thr, is_left in rule:
                if is_left:
                    fire &= X[:, feat] <= thr
                else:
                    fire &= X[:, feat] > thr
            M[:, j] = fire
        return M

    def fit(self, X, y):
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf, max_features=0.5,
            n_jobs=self.n_jobs, random_state=self.seed)
        et = ExtraTreesClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf, max_features=0.5,
            n_jobs=self.n_jobs, random_state=self.seed + 1)
        rf.fit(X, y)
        et.fit(X, y)
        self.rules = self._extract_rules([rf, et])
        R = self._fire(X, self.rules)
        XR = np.hstack([X, R]).astype(np.float32)
        self.scaler = StandardScaler().fit(XR)
        XRs = self.scaler.transform(XR)
        self.lin = SGDClassifier(loss="log_loss", penalty="elasticnet", alpha=self.alpha,
                                 l1_ratio=self.l1_ratio, max_iter=500, tol=1e-4,
                                 random_state=self.seed)
        self.lin.fit(XRs, y)
        return self

    def predict_proba(self, X):
        R = self._fire(X, self.rules)
        XR = np.hstack([X, R]).astype(np.float32)
        XRs = self.scaler.transform(XR)
        return self.lin.predict_proba(XRs)[:, 1]
