import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from scipy.optimize import differential_evolution, LinearConstraint
from tqdm import tqdm
from typing import List

from data_types import Recording, SensorType, get_model_recording
from step_detection import DataHandler
from metric_analysis import AnalysisController


def parse_args(x, round_digits=None) -> dict:
    """Parses the parameters from the optimization result."""
    param_names = [
        'window_duration',
        'min_signal',
        'min_step_delta',
        'max_step_delta',
        'c1', 'c2', 'c3', 'c4',
        'u1', 'u2', 'u3', 'u4',
        'ra', 'r2', 'r3', 'r4',
    ]
    param_map = {
        'confirm_coefs': ['c1', 'c2', 'c3', 'c4'],
        'unconfirm_coefs': ['u1', 'u2', 'u3', 'u4'],
        'reset_coefs': ['ra', 'r2', 'r3', 'r4'],
    }
    params = dict(zip(param_names, x))
    for k, v in param_map.items():
        params[k] = [params[p] for p in v]
        for el in v:
            params.pop(el)
    if round_digits is not None:
        params = {k: np.round(v, round_digits) for k, v in params.items()}
    return params


def get_loss(err: pd.DataFrame, missing_punish_factor=10) -> float:
    all_metrics_represented = err.notna().any(axis=0).all()
    if all_metrics_represented:
        num_missing_rows = err['step_count'].isna().sum()
        missing_data_loss = num_missing_rows*(missing_punish_factor + err.max(axis=None))
        err.fillna(missing_data_loss, inplace=True) # Punish missing data
        loss = err.mean(axis=None)
        return loss
    return np.inf


def optimize_step_detection(datasets: List[Recording], model=None, sensor_type=None, logger=None, maxiter=20, popsize=15) -> dict:
    """Optimizes step detection parameters using differential evolution algorithm."""
    if not len(datasets):
        raise ValueError("No datasets provided.")
    pbar = tqdm(total=(maxiter + 1) * popsize * 10)
    best_current_loss = np.inf
    if model is None:
        model = get_model_recording(datasets[0].sensor_type if sensor_type is None else sensor_type)
    if logger is None:
        logger = AnalysisController.init_logger('optimize.log')
        logger.setLevel('INFO')

    def objective_function(x):
        nonlocal best_current_loss
        params = parse_args(x)
        if np.any(np.isnan(x)):
            logger.warning(f"Invalid parameters: {params}")
            return np.inf
        logger.info(f'Running with parameters: {params}')
        ctrl = AnalysisController(model, logger=logger, **params)
        pbar.update(1)
        measured, truth, _ = ctrl.get_metrics(datasets)
        err = measured.error(truth)
        loss = get_loss(err)
        if loss <= best_current_loss:
            best_current_loss = loss
            pbar.set_description(f'Best Loss: {loss:.1f}')
        if loss != np.inf:
            logger.info(f'Loss: {loss}')
        return loss

    min_window, max_window = 0.05, 0.5
    bounds = [
        (min_window, max_window), # window_duration
        (1e-9, 1),  # min_signal
        (0.01, 2),  # min_step_delta
        (0.01, 2),  # max_step_delta
        *([(1e-9, 2)] * 12),  # 3x4 threshold coefs
    ]
    res = differential_evolution(
        objective_function,
        bounds,
        maxiter=maxiter,
        popsize=popsize,
        constraints=[
            LinearConstraint([1] + [0] * 15, min_window, max_window, keep_feasible=True), # Window duration must be enforced
            LinearConstraint([0] * 2 + [-1, 1] + [0] * 12, lb=0, keep_feasible=True), # min_step_delta < max_step_delta
            LinearConstraint(np.eye(len(bounds)), lb=np.zeros(len(bounds)), keep_feasible=True), # All values must be positive
        ],
    )
    pbar.close()
    logger.info(f'Optimization result: {res}')
    params = parse_args(res.x)
    logger.info(f'Optimized parameters: {params}')
    return res.x

def kfold_optimize(datasets: List[Recording], sensor_type=None, splits=5, seed=0, **kwargs):
    if len(datasets) < splits:
        raise ValueError("Number of datasets must be greater than the number of splits.")
    kf = KFold(splits, shuffle=True, random_state=seed)
    logger = AnalysisController.init_logger('optimize.log')
    logger.setLevel('INFO')
    model = get_model_recording(datasets[0].sensor_type if sensor_type is None else sensor_type)
    X = np.asarray(datasets)

    final_params = []
    final_losses = []
    for train_index, test_index in kf.split(X):
        training_set = X[train_index]
        test_set = X[test_index]
        logger.info(f"Training size: {len(train_index)}, test size: {len(test_index)}")
        params = optimize_step_detection(training_set, model, logger, **kwargs)
        params = parse_args(params)
        ctrl = AnalysisController(model, logger=logger, **params)
        measured, truth, _ = ctrl.get_metrics(test_set)
        err = measured.error(truth)
        loss = get_loss(err)
        final_params.append(params)
        final_losses.append(loss)

    logger.info(f"Losses = {final_losses}")
    logger.info(f"Params = {final_params}")
    best_i = np.argmin(final_losses)
    best_params = final_params[best_i]
    logger.info(f"Best Params: {best_params}")
    return best_params

if __name__ == '__main__':
    datasets = DataHandler.from_sensor_type(SensorType.PIEZO).get(user='ron', location='Aarons Studio')
    optimize_step_detection(datasets, maxiter=10, popsize=10)
    # print(kfold_optimize(datasets, splits=5, maxiter=10))
