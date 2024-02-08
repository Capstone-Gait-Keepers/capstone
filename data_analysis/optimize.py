import numpy as np

from scipy.optimize import differential_evolution, LinearConstraint
from tqdm import tqdm
from typing import List

from data_types import Recording
from step_detection import DataHandler
from metric_analysis import AnalysisController


def optimize_step_detection(datasets: List[Recording], maxiter=2, popsize=15):
    # TODO: Cross fold validation (sklearn.model_selection.StratifiedKFold)
    model_data = Recording.from_file('datasets/2023-11-09_18-42-33.yaml')
    logger = AnalysisController.init_logger('optimize.log')
    pbar = tqdm(total=(maxiter + 1) * popsize * 214)
    def objective_function(x):
        logger.info(f'Running with parameters: {x}')
        window_duration, min_signal, min_step_delta, max_step_delta, ca, cb, cc, cd, ua, ub, uc, ud, ra, rb, rc, rd = x
        ctrl = AnalysisController(
            model_data,
            window_duration=window_duration,
            min_signal=min_signal,
            min_step_delta=min_step_delta,
            max_step_delta=max_step_delta,
            confirm_coefs=(ca, cb, cc, cd),
            unconfirm_coefs=(ua, ub, uc, ud),
            reset_coefs=(ra, rb, rc, rd),
            logger=logger,
        )
        measured, truth, alg = ctrl.get_metrics(datasets)
        pbar.update(1)
        err = measured.error(truth)
        if err['step_count'].notna().all():
            return err.mean().mean()
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
        constraints=[
            # LinearConstraint([1] + [0] * 15, lb=0.05, ub=0.5, keep_feasible=True), # Window duration must be enforced
            LinearConstraint([0] * 2 + [-1, 1] + [0] * 12, lb=0, keep_feasible=True), # min_step_delta < max_step_delta
            LinearConstraint(np.eye(len(bounds)), lb=np.zeros(len(bounds)), keep_feasible=True), # All values must be positive
        #     LinearConstraint(np.hstack([np.zeros((4, 4)), np.eye(4), -np.eye(4), np.zeros((4, 4))]), lb=np.zeros(4)), # Confirm coefs must be large than unconfirm coefs
        #     LinearConstraint(np.hstack([np.zeros((4, 4)), np.eye(4), np.zeros((4, 4)), -np.eye(4)]), lb=np.zeros(4)), # Confirm coefs must be large than reset coefs
        ],
    )
    pbar.close()
    logger.info(f'Optimization result: {res}')
    return res.x


if __name__ == '__main__':
    datasets = DataHandler().get(user='ron', location='Aarons Studio')
    print(optimize_step_detection(datasets))

    # from step_detection import ParsedRecording
    # model = ParsedRecording.from_file('datasets/2023-11-09_18-42-33.yaml')
    # print(model.get_frequency_weights(0.3731458248292704))