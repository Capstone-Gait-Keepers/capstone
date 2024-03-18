import numpy as np
from datetime import datetime, timedelta
from typing import Optional

from data_types import Metrics

np.random.seed(0)


def get_timestamps(count: int, cadence=2, asymmetry=0, var=0) -> np.array:
    timestamps = np.arange(0, count/cadence - 0.0001, 1/cadence) + np.random.normal(0, var, count)
    timestamps[1::2] += asymmetry/cadence
    return timestamps


def generate_metrics(days: int, current_date: Optional[datetime]=None, asymmetry=0, cadence=2, var=0.01) -> Metrics:
    if current_date is None:
        current_date = datetime.now()

    date_range = [current_date - timedelta(days=i) for i in range(days)]
    date_range = [date.strftime("%Y-%m-%d") for date in date_range]

    step_counts = [round(20 + np.random.random() * 5) for _ in range(days)]
    if isinstance(asymmetry, (int, float)):
        asymmetry = [asymmetry] * days
    if isinstance(cadence, (int, float)):
        cadence = [cadence] * days
    if isinstance(var, (int, float)):
        var = [var] * days
    metric_arr = []
    for i in range(days):
        steps = get_timestamps(step_counts[i], cadence=cadence[i], asymmetry=asymmetry[i], var=var[i])
        metric_arr.append(Metrics([steps]))
    metrics: Metrics = sum(metric_arr)
    metrics.set_index(date_range[::-1])
    metrics._df = metrics._df[metrics._df['step_count'] > 2]
    return metrics


def decay(count: int, start: float, end: float):
    np.exp(np.linspace(np.log(start), np.log(end), count))
    steps = np.logspace(0.01, 1, count)
    steps -= steps[0]
    steps *= (end - start)/steps[-1]
    steps += start - steps[0]
    return steps

if __name__ == "__main__":
    # asymmetries = np.logspace(0.01, 1, 5)
    # asymmetries /= 2 * asymmetries.max()
    # for asymmetry in asymmetries:
    #     print(asymmetry)
    #     steps = get_timestamps(5, asymmetry=asymmetry / 2, var=0)
    #     print('steps', steps)

    days = 5
    asymmetry = decay(days, 0, 0)
    cadence = decay(days, 2, 0.5)
    var = decay(days, 0, 0)
    print(generate_metrics(days=days, cadence=cadence, asymmetry=asymmetry, var=var))
