import numpy as np
import pandas as pd

from scipy.optimize import differential_evolution, LinearConstraint
from tqdm import tqdm
from typing import List

from data_types import Recording, get_model_recording
from metrics import Metrics
from step_detection import DataHandler
from metric_analysis import AnalysisController


def parse_analysis_params(x) -> dict:
    """Parses the parameters from the optimization result."""
    window, min_step, max_step, c1, c2, c3, r3, u_scalar, r_scalar = x

    return {
        'window_duration': window,
        'min_step_delta': min_step,
        'max_step_delta': max_step,
        'confirm_coefs': [c1, c2, 0, c3],
        'unconfirm_coefs': [c1*u_scalar, c2*u_scalar, 0, c3*u_scalar],
        'reset_coefs': [c1*r_scalar, c2*r_scalar, 0, r3],
    }


def get_init_params(popsize: int, var=5) -> np.ndarray:
    """Generates initial population for the optimization around a known set of parameters."""
    defaults = [
        0.15, # window_duration
        0.4,  # min_step_delta
        1,  # max_step_delta
        0.07, 0, 0.012, 0.012, 4/7, 0, # coefs
    ]
    # Vary defaults by var to generate initial population of size popsize
    cases = [defaults]
    for _ in range(popsize - 1):
        case = [d + np.random.uniform(-var, var) for d in defaults]
        cases.append(case)
    return np.array(cases)


def metric_loss(err: pd.DataFrame, missing_punish_factor=10) -> float:
    """Loss based on metric errors. Punishes missing data."""
    all_metrics_represented = err.notna().any(axis=0).all()
    if all_metrics_represented:
        num_missing_rows = err['step_count'].isna().sum()
        missing_data_loss = num_missing_rows*(missing_punish_factor + err.max(axis=None))
        err.fillna(missing_data_loss, inplace=True) # Punish missing data
        loss = err.mean(axis=None)
        return loss
    return np.inf


def false_rate_loss(false_neg: float, false_pos: float, falsity_weight_ratio=1):
    """Loss based on false negative and false positive rates."""
    return np.average([false_neg, false_pos], weights=[falsity_weight_ratio, 1])


def optimize_step_detection(datasets: List[Recording], model=None, sensor_type=None, fs=None, logger=None, maxiter=20, popsize=15, truthful_steps_limit=4, step_delta_min_range=0.3) -> dict:
    """
    Optimizes step detection parameters using differential evolution algorithm.

    Parameters
    ----------
    datasets : List[Recording]
        List of recordings to use for optimization.
    model : Recording, optional
        Model to use for step detection. If not provided, a model will be found based on the sensor type.
    sensor_type : SensorType, optional
        Sensor type to use for step detection. If not provided, the sensor type of the first Recording in
        the dataset will be used.
    logger : Logger, optional
        Logger to use for logging. If not provided, a logger will be initialized.
    maxiter : int, optional
        Maximum number of iterations for the optimization algorithm.
    popsize : int, optional
        Population size for the optimization algorithm.
    step_delta_min_range : float, optional
        Minimum range between min_step_delta and max_step_delta to constrain the optimization.
    falsity_weight_ratio : float, optional
        Weight ratio for false negative and false positive rates. Typically used to skew the optimization
        towards minimizing false negatives.
    """
    if not len(datasets):
        raise ValueError("No datasets provided.")
    pbar = tqdm(total=(maxiter + 1) * popsize * 5)
    best_current_loss = np.inf
    if sensor_type is None:
        sensor_type = datasets[0].sensor_type
    if fs is None:
        fs = datasets[0].env.fs
    if model is None:
        model = get_model_recording(sensor_type, fs=fs)
    if logger is None:
        logger = AnalysisController.init_logger('optimize.log')
        logger.setLevel('INFO')

    def objective_function(x):
        nonlocal best_current_loss
        params = parse_analysis_params(x)
        if np.any(np.isnan(x)):
            logger.warning(f"Invalid parameters: {params}")
            return np.inf
        logger.info(f'Running with parameters: {params}')
        ctrl = AnalysisController(model, logger=logger, **params)
        pbar.update(1)
        measured, truth, _ = ctrl.get_metrics(datasets)
        error = measured.error(truth)
        metrics = [col for col in error.columns if col in Metrics.STEP_REQUIREMENTS and Metrics.STEP_REQUIREMENTS[col] <= truthful_steps_limit]
        error = error.filter(metrics)
        loss = metric_loss(error)
        if loss <= best_current_loss:
            best_current_loss = loss
            pbar.set_description(f'Loss={loss:.4f}')
        logger.info(f'Loss={loss:.4f}')
        return loss

    min_window, max_window = 0.05, 0.2
    bounds = [
        (min_window, max_window), # window_duration
        # (0, 0.05),  # min_signal
        (0.1, 2 - step_delta_min_range),  # min_step_delta
        (0.1 + step_delta_min_range, 2),  # max_step_delta
        *([(0, 2)] * 4),  # threshold coefs
        *([(0, 1)] * 2),  # u & r relative threshold coefs
    ]
    init = get_init_params(popsize)
    res = differential_evolution(
        objective_function,
        bounds,
        maxiter=maxiter,
        popsize=popsize,
        init=init,
        constraints=[
            LinearConstraint([1] + [0] * (len(bounds) - 1), min_window, max_window, keep_feasible=True), # Window duration must be enforced
            LinearConstraint([0] + [-1, 1] + [0] * 6, lb=step_delta_min_range, keep_feasible=True), # max_step_delta - min_step_delta > step_delta_min_range
            LinearConstraint(np.eye(len(bounds)), lb=np.zeros(len(bounds)), keep_feasible=True), # All values must be positive
        ],
        polish=False,
    )
    pbar.close()
    logger.info(f'Optimization result: {res}')
    params = parse_analysis_params(res.x)
    logger.info(f'Optimized parameters: {params}')
    return params


if __name__ == '__main__':
    print(optimize_step_detection(DataHandler('datasets/piezo_custom').get(), maxiter=50, popsize=100))

    # datasets = DataHandler.from_sensor_type(SensorType.ACCEL).get(user='ron', quality='normal', location='Aarons Studio')
    # # datasets = [d for d in datasets if d.env.quality in ('chaotic', 'normal')]
    # print(optimize_step_detection(datasets, maxiter=20, popsize=50))
