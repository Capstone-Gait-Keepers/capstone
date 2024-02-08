import numpy as np

from scipy.optimize import differential_evolution, LinearConstraint
from tqdm import tqdm
from typing import List

from data_types import Recording
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


def optimize_step_detection(datasets: List[Recording], missing_punish_factor=10, maxiter=3, popsize=15) -> dict:
    """Optimizes step detection parameters using differential evolution algorithm."""
    # TODO: Cross fold validation (sklearn.model_selection.StratifiedKFold)
    model_data = Recording.from_file('datasets/2023-11-09_18-42-33.yaml')
    logger = AnalysisController.init_logger('optimize.log')
    logger.setLevel('INFO')
    pbar = tqdm(total=(maxiter + 1) * popsize * 10)

    def objective_function(x):
        params = parse_args(x)
        if np.any(x < 0) or np.any(np.isnan(x)):
            logger.warning(f"Invalid parameters: {params}")
            return np.inf
        logger.info(f'Running with parameters: {parse_args(x, round_digits=4)}')
        if params['min_step_delta'] > params['max_step_delta']:
            return np.inf
        ctrl = AnalysisController(model_data, logger=logger, **params)
        measured, truth, alg = ctrl.get_metrics(datasets)
        pbar.update(1)
        err = measured.error(truth)
        all_metrics_represented = err.notna().any(axis=0).all()
        if all_metrics_represented:
            num_missing_rows = err['step_count'].isna().sum()
            missing_data_loss = num_missing_rows*(missing_punish_factor + err.max(axis=None))
            err.fillna(missing_data_loss, inplace=True) # Punish missing data
            loss = err.mean(axis=None)
            logger.info(f'Cumulative Error: {loss}')
            return loss
        return np.inf

    bounds = [
        (0.05, 0.5), # window_duration
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
    )
    pbar.close()
    logger.info(f'Optimization result: {res}')
    params = parse_args(res.x)
    logger.info(f'Optimized parameters: {params}')
    return params


if __name__ == '__main__':
    datasets = DataHandler().get(user='ron', location='Aarons Studio')
    print(optimize_step_detection(datasets))

    # from step_detection import ParsedRecording
    # model = ParsedRecording.from_file('datasets/2023-11-09_18-42-33.yaml')
    # print(model.get_frequency_weights(0.3731458248292704))
