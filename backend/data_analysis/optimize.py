import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from scipy.optimize import differential_evolution, LinearConstraint
from tqdm import tqdm
from typing import List

from data_types import Recording, SensorType, get_model_recording
from step_detection import DataHandler
from metric_analysis import AnalysisController


def parse_analysis_params(x, round_digits=None) -> dict:
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


def get_init_params(popsize: int, var=1) -> np.ndarray:
    """Generates initial population for the optimization around a known set of parameters."""
    defaults = [
        0.2, # window_duration
        0.05,  # min_signal
        0.1,  # min_step_delta
        2,  # max_step_delta
        *(0.5, 0.3, 0, 0), # confirmed
        *(0.25, 0.65, 0, 0), # unconfirmed
        *(0, 1, 0, 0), # reset
    ]
    # Vary defaults by var to generate initial population of size popsize
    cases = []
    for _ in range(popsize):
        case = [d + np.random.uniform(-var, var) for d in defaults]
        cases.append(case)
    return np.array(cases)



def optimize_step_detection(datasets: List[Recording], model=None, sensor_type=None, logger=None, maxiter=20, popsize=15, step_delta_min_range=0.3, falsity_weight_ratio=1) -> dict:
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
    pbar = tqdm(total=(maxiter + 1) * popsize * 10)
    best_current_loss = np.inf
    if model is None:
        model = get_model_recording(datasets[0].sensor_type if sensor_type is None else sensor_type)
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
        false_neg, false_pos = ctrl.get_false_rates(datasets)
        loss = np.average([false_neg, false_pos], weights=[falsity_weight_ratio, 1])
        if loss <= best_current_loss:
            best_current_loss = loss
            pbar.set_description(f'FNR={false_neg:.2f}, FPR={false_pos:.2f}')
        logger.info(f'FNR={false_neg:.2f}, FPR={false_pos:.2f}, Loss={loss:.2f}')
        return loss

    min_window, max_window = 0.05, 0.5
    bounds = [
        (min_window, max_window), # window_duration
        (0, 0.2),  # min_signal
        (0, 2),  # min_step_delta
        (0, 2),  # max_step_delta
        *([(0, 2)] * 12),  # 3x4 threshold coefs
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
            LinearConstraint([0] * 2 + [-1, 1] + [0] * 12, lb=step_delta_min_range, keep_feasible=True), # max_step_delta - min_step_delta > step_delta_min_range
            LinearConstraint(np.eye(len(bounds)), lb=np.zeros(len(bounds)), keep_feasible=True), # All values must be positive
        ],
    )
    pbar.close()
    logger.info(f'Optimization result: {res}')
    params = parse_analysis_params(res.x)
    logger.info(f'Optimized parameters: {params}')
    return params


if __name__ == '__main__':
    datasets = DataHandler.from_sensor_type(SensorType.PIEZO).get(user='ron', session='2024-02-11', location='Aarons Studio')
    datasets = [d for d in datasets if d.env.quality in ('chaotic', 'normal')]
    print(optimize_step_detection(datasets, maxiter=50))
