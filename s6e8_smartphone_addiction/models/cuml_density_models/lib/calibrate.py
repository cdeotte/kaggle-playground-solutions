"""OOF / test probabilities from the fold outputs of the SVC / RFF / RF models.

The 5 fold models are not on a common scale (the SVC outputs a decision function), so:
  1. per-fold common scale: each fold model's validation outputs -> percentile ranks within its validation rows; its test
     outputs -> percentile ranks within the test set;
  2. OOF: cross-fitted isotonic regression rank -> label: the rows of fold f are mapped by an isotonic fit on the ranks and
     labels of the other 4 folds;
  3. test: the 5 test rank vectors are averaged and re-ranked, then mapped by an isotonic fit on all OOF ranks.
Both sides are clipped to [1e-6, 1 - 1e-6]; + 1e-9 x rank restores the model's order inside the isotonic plateaus."""
import numpy as np
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from common.data import folds as fold_ids, target
from common.pairs import save_pair

EPS = 1e-6


def calibrate(model_dir, pair):
    y = target()
    folds = fold_ids(y)
    r_oof, tst_ranks = np.zeros(len(y)), []
    for f in range(5):
        pv = np.load(model_dir / 'parts' / f'{pair}_f{f}_valid.npy')
        pt = np.load(model_dir / 'parts' / f'{pair}_f{f}_test.npy')
        m = folds == f
        r_oof[m] = rankdata(pv) / m.sum()
        tst_ranks.append(rankdata(pt) / len(pt))

    p_oof = np.zeros(len(y))
    for f in range(5):
        fit, app = folds != f, folds == f
        ir = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        ir.fit(r_oof[fit], y[fit])
        p_oof[app] = ir.predict(r_oof[app])
    p_oof = np.clip(p_oof, EPS, 1 - EPS) + 1e-9 * r_oof

    r_tst = rankdata(np.column_stack(tst_ranks).mean(axis=1)) / len(tst_ranks[0])
    ir_all = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    ir_all.fit(r_oof, y)
    p_tst = np.clip(ir_all.predict(r_tst), EPS, 1 - EPS) + 1e-9 * r_tst

    save_pair(model_dir, pair, p_oof, p_tst)
    print(f'{pair}: OOF AUC {roc_auc_score(y, p_oof):.6f} (ranks {roc_auc_score(y, r_oof):.6f}), '
          f'OOF mean {p_oof.mean():.4f}, test mean {p_tst.mean():.4f}', flush=True)
