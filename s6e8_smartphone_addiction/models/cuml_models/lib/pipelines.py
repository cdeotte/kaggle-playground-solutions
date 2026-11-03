"""The six fold pipelines. Every fitted state (scalers, medians, PCA, KMeans, UMAP, bandwidths, models) uses the
outer-train rows of the fold only. Each pipeline takes base = (train, valid, test) feature arrays, the response bank
response = {'train', 'valid', 'test'}, the outer-train labels y and the fold number, and returns the held-out and test
probabilities."""
import numpy as np
from scipy.special import expit


def even_rows(n, rows):
    """`rows` evenly spaced row positions out of n (all rows if n <= rows)."""
    return np.linspace(0, n - 1, min(rows, n), dtype=np.int64)


def robust_scale(train, valid, test):
    """Median fill, (x - median) / IQR, clip +-12; median and IQR from 120,000 evenly spaced outer-train rows."""
    sample = train[even_rows(len(train), 120_000)].astype(np.float64, copy=False)
    median = np.nanmedian(sample, axis=0)
    q25 = np.nanquantile(sample, 0.25, axis=0)
    q75 = np.nanquantile(sample, 0.75, axis=0)
    scale = q75 - q25
    median[~np.isfinite(median)] = 0.0
    scale[~np.isfinite(scale) | (scale < 1e-5)] = 1.0

    def apply(value):
        output = np.asarray(value, np.float32).copy()
        missing = ~np.isfinite(output)
        if missing.any():
            output[missing] = np.broadcast_to(median, output.shape)[missing]
        output = (output - median) / scale
        return np.clip(output, -12.0, 12.0).astype(np.float32, copy=False)
    return apply(train), apply(valid), apply(test)


def median_fill(train, valid, test):
    """Median fill (120,000 evenly spaced outer-train rows), clip +-1e5."""
    median = np.nanmedian(train[even_rows(len(train), 120_000)].astype(np.float64), axis=0)
    median[~np.isfinite(median)] = 0.0

    def apply(value):
        output = np.asarray(value, np.float32).copy()
        missing = ~np.isfinite(output)
        if missing.any():
            output[missing] = np.broadcast_to(median, output.shape)[missing]
        return np.clip(output, -1e5, 1e5).astype(np.float32, copy=False)
    return apply(train), apply(valid), apply(test)


def pca_transform(train, valid, test, components):
    """Whitened cuML PCA (full SVD) fitted on all outer-train rows."""
    import cupy as cp
    from cuml.decomposition import PCA
    model = PCA(n_components=components, whiten=True, svd_solver='full', output_type='cupy')
    train_out = cp.asnumpy(model.fit_transform(cp.asarray(train))).astype(np.float32)
    valid_out = cp.asnumpy(model.transform(cp.asarray(valid))).astype(np.float32)
    test_out = cp.asnumpy(model.transform(cp.asarray(test))).astype(np.float32)
    print(f'  PCA-{components}: explained variance {float(cp.asnumpy(model.explained_variance_ratio_).sum()):.4f}',
          flush=True)
    return train_out, valid_out, test_out


def fit_lr(train, y, valid, test, C):
    """cuML L2 logistic regression (qn, max_iter 1200, tol 1e-6)."""
    import cupy as cp
    from cuml.linear_model import LogisticRegression
    model = LogisticRegression(C=C, penalty='l2', max_iter=1200, tol=1e-6, solver='qn', output_type='cupy')
    model.fit(cp.asarray(train), cp.asarray(y))
    valid_prediction = cp.asnumpy(model.predict_proba(cp.asarray(valid))[:, 1])
    test_prediction = cp.asnumpy(model.predict_proba(cp.asarray(test))[:, 1])
    return valid_prediction.astype(np.float64), test_prediction.astype(np.float64)


def stack_columns(*blocks):
    return np.column_stack(blocks).astype(np.float32)


def original_knn_lr(base, response, y, fold):
    """Robust-scaled base (168 columns, no observed-joint block) + robust-scaled response bank -> LR, C 0.12."""
    train, valid, test = robust_scale(*base)
    rtrain, rvalid, rtest = robust_scale(response['train'], response['valid'], response['test'])
    return fit_lr(stack_columns(train, rtrain), y, stack_columns(valid, rvalid), stack_columns(test, rtest), C=0.12)


def residual_pca_svm(base, response, y, fold):
    """Robust-scaled base -> whitened PCA-64, + robust-scaled response bank -> squared-hinge linear SVM (C 0.035);
    probability = sigmoid(decision function)."""
    import cupy as cp
    from cuml.svm import LinearSVC
    train, valid, test = robust_scale(*base)
    ptrain, pvalid, ptest = pca_transform(train, valid, test, 64)
    rtrain, rvalid, rtest = robust_scale(response['train'], response['valid'], response['test'])
    model = LinearSVC(C=0.035, penalty='l2', loss='squared_hinge', max_iter=1500, tol=1e-6, output_type='cupy')
    model.fit(cp.asarray(stack_columns(ptrain, rtrain)), cp.asarray(y))
    valid_score = cp.asnumpy(model.decision_function(cp.asarray(stack_columns(pvalid, rvalid))))
    test_score = cp.asnumpy(model.decision_function(cp.asarray(stack_columns(ptest, rtest))))
    return expit(valid_score).astype(np.float64), expit(test_score).astype(np.float64)


def rbf_map(ptrain, pvalid, ptest, seed):
    """256 cuML KMeans landmarks (180,000 evenly spaced outer-train rows); features exp(-d^2 / (2 (f sigma)^2)) for
    f in (0.65, 1, 1.75), sigma = median nearest-landmark distance of the fit rows."""
    import cupy as cp
    from cuml.cluster import KMeans
    positions = even_rows(len(ptrain), 180_000)
    model = KMeans(n_clusters=256, max_iter=120, tol=1e-4, random_state=seed, n_init=1, max_samples_per_batch=32768,
                   output_type='cupy')
    model.fit(cp.asarray(ptrain[positions]))
    centers = cp.asarray(model.cluster_centers_, dtype=cp.float32)
    center_sq = cp.sum(cp.square(centers), axis=1)[None, :]

    def distance(value, chunk=80_000):
        blocks = []
        for start in range(0, len(value), chunk):
            query = cp.asarray(value[start:start + chunk])
            d2 = cp.sum(cp.square(query), axis=1)[:, None] + center_sq - cp.float32(2.0) * query @ centers.T
            blocks.append(cp.asnumpy(cp.maximum(d2, 0.0)).astype(np.float32))
        return np.concatenate(blocks, axis=0)

    train_d2, valid_d2, test_d2 = distance(ptrain), distance(pvalid), distance(ptest)
    nearest = np.sqrt(np.min(train_d2[positions], axis=1))
    sigma = float(np.median(nearest[nearest > 1e-6]))
    if not np.isfinite(sigma) or sigma <= 0:
        raise RuntimeError('landmark bandwidth is invalid')
    print(f'  landmark sigma {sigma:.4f}', flush=True)

    def apply(d2):
        return np.column_stack([np.exp(-d2 / (2.0 * (sigma * factor) ** 2)) for factor in (0.65, 1.0, 1.75)]).astype(np.float32)
    return apply(train_d2), apply(valid_d2), apply(test_d2)


def landmark_rbf_lr(base, response, y, fold):
    """Robust-scaled base -> whitened PCA-40 -> 768 landmark RBF features; [PCA, RBF, response] -> LR, C 0.08."""
    train, valid, test = robust_scale(*base)
    ptrain, pvalid, ptest = pca_transform(train, valid, test, 40)
    btrain, bvalid, btest = rbf_map(ptrain, pvalid, ptest, 2600 + fold)
    rtrain, rvalid, rtest = robust_scale(response['train'], response['valid'], response['test'])
    return fit_lr(stack_columns(ptrain, btrain, rtrain), y, stack_columns(pvalid, bvalid, rvalid),
                  stack_columns(ptest, btest, rtest), C=0.08)


def batched_umap_transform(model, value, chunk=80_000):
    import cupy as cp
    output = []
    for start in range(0, len(value), chunk):
        output.append(cp.asnumpy(model.transform(cp.asarray(value[start:start + chunk]))).astype(np.float32, copy=False))
    return np.concatenate(output, axis=0)


def umap_lr(base, response, y, fold):
    """Robust-scaled base -> cuML UMAP-16 fitted on 120,000 evenly spaced outer-train rows (40 neighbors, 240 epochs,
    min_dist 0.08, nn-descent) -> robust-scaled embedding + response bank -> LR, C 0.16."""
    import cupy as cp
    from cuml.manifold import UMAP
    train, valid, test = robust_scale(*base)
    model = UMAP(n_neighbors=40, n_components=16, n_epochs=240, learning_rate=1.0, min_dist=0.08, spread=1.0,
                 metric='euclidean', random_state=3600 + fold, build_algo='nn_descent', output_type='cupy', verbose=False)
    model.fit(cp.asarray(train[even_rows(len(train), 120_000)]))
    utrain, uvalid, utest = robust_scale(batched_umap_transform(model, train), batched_umap_transform(model, valid),
                                         batched_umap_transform(model, test))
    rtrain, rvalid, rtest = robust_scale(response['train'], response['valid'], response['test'])
    return fit_lr(stack_columns(utrain, rtrain), y, stack_columns(uvalid, rvalid), stack_columns(utest, rtest), C=0.16)


def artifact_rf(base, response, y, fold):
    """Median-filled 240 target-free base columns (no response bank) -> 900-tree cuML random forest."""
    import cupy as cp
    from cuml.ensemble import RandomForestClassifier
    train, valid, test = median_fill(*base)
    model = RandomForestClassifier(n_estimators=900, max_depth=18, max_features=0.55, min_samples_leaf=18,
                                   min_samples_split=36, n_bins=256, bootstrap=True, max_samples=0.78, n_streams=8,
                                   random_state=4600 + fold, output_type='cupy')
    model.fit(cp.asarray(train), cp.asarray(y))
    valid_prediction = cp.asnumpy(model.predict_proba(cp.asarray(valid))[:, 1])
    test_prediction = cp.asnumpy(model.predict_proba(cp.asarray(test))[:, 1])
    return valid_prediction.astype(np.float64), test_prediction.astype(np.float64)


def cluster_features(ptrain, pvalid, ptest, y, fold):
    """For 64, 256 and 1024 cuML KMeans clusters (220,000 evenly spaced outer-train rows): the cluster's smoothed
    (m 25) label logit, log cluster size and RMS distance to the center. Outer-train rows get the exact
    leave-one-row-out posterior; held-out / test rows the posterior of all outer-train rows."""
    import cupy as cp
    from cuml.cluster import KMeans
    columns = {'train': [], 'valid': [], 'test': []}
    prior = float(y.mean())

    def logit(probability):
        probability = np.clip(probability, 1e-5, 1.0 - 1e-5)
        return np.log(probability / (1.0 - probability)).astype(np.float32)

    for resolution in (64, 256, 1024):
        model = KMeans(n_clusters=resolution, max_iter=140, tol=1e-4, random_state=5600 + 10 * fold + resolution,
                       n_init=1, max_samples_per_batch=32768, output_type='cupy')
        model.fit(cp.asarray(ptrain[even_rows(len(ptrain), 220_000)]))
        centers = cp.asnumpy(model.cluster_centers_).astype(np.float32)

        def assign(value, chunk=100_000):
            labels, distances = [], []
            for start in range(0, len(value), chunk):
                query = value[start:start + chunk]
                label = cp.asnumpy(model.predict(cp.asarray(query))).astype(np.int32)
                distances.append(np.sqrt(np.mean(np.square(query - centers[label]), axis=1)).astype(np.float32))
                labels.append(label)
            return np.concatenate(labels), np.concatenate(distances)

        train_label, train_distance = assign(ptrain)
        valid_label, valid_distance = assign(pvalid)
        test_label, test_distance = assign(ptest)
        count = np.bincount(train_label, minlength=resolution).astype(np.float64)
        positive = np.bincount(train_label, weights=y, minlength=resolution).astype(np.float64)
        smoothing = 25.0
        train_probability = (positive[train_label] - y + smoothing * prior) / (count[train_label] - 1.0 + smoothing)
        full_probability = (positive + smoothing * prior) / (count + smoothing)
        columns['train'] += [logit(train_probability), np.log1p(count[train_label]), train_distance]
        columns['valid'] += [logit(full_probability[valid_label]), np.log1p(count[valid_label]), valid_distance]
        columns['test'] += [logit(full_probability[test_label]), np.log1p(count[test_label]), test_distance]
    return tuple(np.column_stack(columns[role]).astype(np.float32) for role in ('train', 'valid', 'test'))


def cluster_posterior_lr(base, response, y, fold):
    """Robust-scaled base -> whitened PCA-32 -> 9 cluster columns (robust-scaled); [PCA, clusters, response] -> LR,
    C 0.12."""
    train, valid, test = robust_scale(*base)
    ptrain, pvalid, ptest = pca_transform(train, valid, test, 32)
    ctrain, cvalid, ctest = cluster_features(ptrain, pvalid, ptest, y, fold)
    rtrain, rvalid, rtest = robust_scale(response['train'], response['valid'], response['test'])
    ctrain, cvalid, ctest = robust_scale(ctrain, cvalid, ctest)
    return fit_lr(stack_columns(ptrain, ctrain, rtrain), y, stack_columns(pvalid, cvalid, rvalid),
                  stack_columns(ptest, ctest, rtest), C=0.12)


PIPELINES = {
    'original_knn_lr': original_knn_lr,
    'residual_pca_svm': residual_pca_svm,
    'landmark_rbf_lr': landmark_rbf_lr,
    'umap_lr': umap_lr,
    'artifact_rf': artifact_rf,
    'cluster_posterior_lr': cluster_posterior_lr,
}
