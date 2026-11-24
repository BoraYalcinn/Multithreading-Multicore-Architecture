"""PC40 Lab 05 -- nested cross-validation, sequential reference and the three
explicit parallel decompositions.

Outer loop : LOOCV over the 25 observations (generalization estimate)
Inner loop : 5-fold CV over the 24 outer-training observations (model selection)
Grid       : 4 x 3 x 3 = 36 decision-tree configurations

The lab forbids GridSearchCV and n_jobs=-1 for the required implementation, so
every loop is written out and every task is created by hand.

Determinism.  The result must not depend on how the work is scheduled, so:
  * the hyperparameter grid is built once, in a fixed order, and indexed;
  * KFold is used with shuffle=False, which gives the 5+5+5+5+4 inner split
    the lab describes, and never depends on a random draw;
  * every tree is built with random_state=42;
  * the reduction over configurations is always done afterwards, in grid order,
    with a strict '<' so the first minimum wins.
A parallel run therefore has to return exactly the same external predictions as
the sequential one -- not merely close ones.
"""

from __future__ import annotations

import functools
import itertools
import os

# Pin the native thread pools BEFORE numpy and scikit-learn are imported: these
# variables are only read when the library initialises, so setting them later
# has no effect.  Without this, BLAS starts one thread per core inside every
# task and the explicit parallelism being measured is buried under threads
# nobody asked for.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

SEED = 42


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------
def make_dataset(n_samples: int = 25, seed: int = SEED):
    """The lab states the problem for 25 observations but supplies no data, so
    the dataset is generated here and the generating function is written down,
    which makes the whole experiment reproducible from this file alone.

        x1, x2, x3 ~ U(0, 1)
        y = 8*x1 + 5*sin(3*x2) - 4*x3^2 + e,     e ~ N(0, 0.5^2)

    The response is deliberately not a step function: a regression tree can only
    approximate it, so a deeper tree fits the training data better while
    generalizing worse.  That is what makes the validation protocol meaningful
    rather than decorative.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n_samples, 3))
    noise = rng.normal(0.0, 0.5, size=n_samples)
    y = 8.0 * X[:, 0] + 5.0 * np.sin(3.0 * X[:, 1]) - 4.0 * X[:, 2] ** 2 + noise
    return X, y


PARAM_GRID = {
    "max_depth": [2, 3, 4, 5],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [1, 2, 3],
}


def build_grid(param_grid=None):
    """The 36 configurations, in one fixed order that every version reuses."""
    grid = param_grid if param_grid is not None else PARAM_GRID
    keys = list(grid.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*grid.values())]


def make_model(params, kind: str = "tree"):
    """The three estimators the lab asks for, all with a fixed random_state.

    XGBoost is given n_jobs=1 on purpose: the lab wants the explicit task
    parallelism measured, and a library that starts its own threads inside each
    task would oversubscribe the machine and make the speedup curve unreadable.
    """
    if kind == "tree":
        return DecisionTreeRegressor(random_state=SEED, **params)
    if kind == "boosting":
        return GradientBoostingRegressor(random_state=SEED, **params)
    if kind == "xgb":
        from xgboost import XGBRegressor                   # optional dependency
        return XGBRegressor(random_state=SEED, n_jobs=1, verbosity=0, **params)
    raise ValueError(f"unknown model kind: {kind}")


BOOSTING_GRID = {
    "n_estimators": [50, 100],
    "learning_rate": [0.05, 0.1],
    "max_depth": [2, 3],
}

XGB_GRID = dict(BOOSTING_GRID)


# --------------------------------------------------------------------------
# the two primitives every version is built from
# --------------------------------------------------------------------------
def inner_fold_error(X, y, train_idx, val_idx, params, kind="tree"):
    """One model fit and one validation score: the smallest unit of work."""
    model = make_model(params, kind)
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[val_idx])
    return float(np.mean(np.abs(y[val_idx] - pred)))          # MAE on the fold


def inner_splits(outer_train_idx, n_splits=5):
    """The 5 inner folds of one outer iteration, as index arrays.

    shuffle=False, so for 24 observations this is exactly the 5+5+5+5+4 split
    of the lab brief and is identical in every run and every worker.
    """
    kf = KFold(n_splits=n_splits, shuffle=False)
    return [(outer_train_idx[tr], outer_train_idx[va])
            for tr, va in kf.split(outer_train_idx)]


def mean_inner_score(X, y, splits, params, kind="tree"):
    """Mean validation error of one configuration over the inner folds."""
    return float(np.mean([inner_fold_error(X, y, tr, va, params, kind)
                          for tr, va in splits]))


def select_best(scores, grid):
    """The reduction: minimum validation error, grid order, first one wins.

    Done by one thread after the tasks have finished, never by workers racing
    on a shared best-so-far variable.
    """
    best_i, best_score = 0, float("inf")
    for i, score in enumerate(scores):
        if score < best_score:
            best_i, best_score = i, score
    return best_i, best_score, grid[best_i]


# --------------------------------------------------------------------------
# one outer iteration -- the body shared by the sequential and parallel runs
# --------------------------------------------------------------------------
def outer_iteration(i, X, y, grid, kind="tree", inner_pool=None):
    """Hold out observation i, select hyperparameters on the other 24, refit on
    all 24 and predict the held-out one.

    `inner_pool` selects decomposition C: when it is given, the 5 inner folds of
    each configuration are submitted to it instead of being looped over.
    """
    n = X.shape[0]
    outer_train_idx = np.array([j for j in range(n) if j != i])
    splits = inner_splits(outer_train_idx)

    scores = []
    for params in grid:
        if inner_pool is None:
            scores.append(mean_inner_score(X, y, splits, params, kind))
        else:
            task = functools.partial(_fold_task, X=X, y=y, params=params,
                                     kind=kind)
            fold_errors = list(inner_pool.map(task, splits))
            scores.append(float(np.mean(fold_errors)))

    _, best_score, best_params = select_best(scores, grid)

    final = make_model(best_params, kind)
    final.fit(X[outer_train_idx], y[outer_train_idx])
    prediction = float(final.predict(X[i:i + 1])[0])
    return {"index": i, "y_true": float(y[i]), "y_pred": prediction,
            "best_params": best_params, "best_inner_score": best_score}


def outer_iteration_hyper_parallel(i, X, y, grid, pool, kind="tree"):
    """Decomposition B: inside one outer iteration, one task per configuration.

    Each task owns one configuration, reads X and y read-only, and returns one
    number.  The parent collects the 36 numbers in grid order and reduces them.
    """
    n = X.shape[0]
    outer_train_idx = np.array([j for j in range(n) if j != i])
    splits = inner_splits(outer_train_idx)

    task = functools.partial(_config_task, X=X, y=y, splits=splits, kind=kind)
    scores = list(pool.map(task, grid))

    _, best_score, best_params = select_best(scores, grid)
    final = make_model(best_params, kind)
    final.fit(X[outer_train_idx], y[outer_train_idx])
    return {"index": i, "y_true": float(y[i]),
            "y_pred": float(final.predict(X[i:i + 1])[0]),
            "best_params": best_params, "best_inner_score": best_score}


# --------------------------------------------------------------------------
# the four runners
# --------------------------------------------------------------------------
def _outer_task(i, X, y, grid, kind):
    """Module-level so that a ProcessPoolExecutor can pickle it; a lambda or a
    closure cannot cross a process boundary."""
    return outer_iteration(i, X, y, grid, kind)


def _config_task(params, X, y, splits, kind):
    return mean_inner_score(X, y, splits, params, kind)


def _fold_task(split, X, y, params, kind):
    return inner_fold_error(X, y, split[0], split[1], params, kind)


def nested_cv_sequential(X, y, grid=None, kind="tree"):
    grid = grid if grid is not None else build_grid()
    return [outer_iteration(i, X, y, grid, kind) for i in range(X.shape[0])]


def nested_cv_outer_parallel(X, y, pool, grid=None, kind="tree"):
    """Decomposition A: one task per outer LOOCV iteration (25 tasks).

    Task i owns output slot i.  Nothing is shared except the read-only data and
    grid, so there is no lock anywhere; the only synchronization is the join
    inside pool.map, which also restores the original order.
    """
    grid = grid if grid is not None else build_grid()
    task = functools.partial(_outer_task, X=X, y=y, grid=grid, kind=kind)
    return list(pool.map(task, range(X.shape[0])))


def nested_cv_hyper_parallel(X, y, pool, grid=None, kind="tree"):
    """Decomposition B: outer loop serial, 36 configuration tasks inside it."""
    grid = grid if grid is not None else build_grid()
    return [outer_iteration_hyper_parallel(i, X, y, grid, pool, kind)
            for i in range(X.shape[0])]


def nested_cv_inner_parallel(X, y, pool, grid=None, kind="tree"):
    """Decomposition C: outer and grid loops serial, 5 fold tasks at the bottom.

    The most tasks and the smallest ones: 25 x 36 x 5 = 4500 submissions.
    """
    grid = grid if grid is not None else build_grid()
    return [outer_iteration(i, X, y, grid, kind, inner_pool=pool)
            for i in range(X.shape[0])]


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def metrics(results):
    y_true = np.array([r["y_true"] for r in results])
    y_pred = np.array([r["y_pred"] for r in results])
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {"MAE": mae, "RMSE": rmse, "R2": 1.0 - ss_res / ss_tot}


def predictions_of(results):
    return np.array([r["y_pred"] for r in results])


# --------------------------------------------------------------------------
# final model: 5-fold CV on all 25 observations, then fit on all 25
# --------------------------------------------------------------------------
def final_model_selection(X, y, grid=None, kind="tree", n_splits=5):
    grid = grid if grid is not None else build_grid()
    all_idx = np.arange(X.shape[0])
    splits = inner_splits(all_idx, n_splits=n_splits)
    scores = [mean_inner_score(X, y, splits, params, kind) for params in grid]
    _, best_score, best_params = select_best(scores, grid)
    model = make_model(best_params, kind)
    model.fit(X, y)
    return model, best_params, best_score


# --------------------------------------------------------------------------
# thread settings
# --------------------------------------------------------------------------
def native_thread_settings():
    """What the native thread pools were actually limited to, for the record.

    The limits themselves are applied at the top of this module, before numpy
    is imported, because that is the only moment at which they are read.
    """
    return {var: os.environ.get(var) for var in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}


def make_pool(kind: str, workers: int):
    return (ThreadPoolExecutor(max_workers=workers) if kind == "thread"
            else ProcessPoolExecutor(max_workers=workers))
