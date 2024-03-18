import numpy as np
from datetime import datetime, timedelta
from typing import Optional

from data_types import Metrics

np.random.seed(0)


def get_timestamps(count: int, cadence=1, var=0.05) -> np.array:
    return np.arange(0, count, cadence) + np.random.normal(0, var, count)


def generate_metrics(days: int, current_date: Optional[datetime]=None) -> Metrics:
    if current_date is None:
        current_date = datetime.now()

    date_range = [current_date - timedelta(days=i) for i in range(days)]
    date_range = [date.strftime("%Y-%m-%d") for date in date_range]

    step_counts = [round(1 + np.random.random() * 5) for _ in range(days)]
    metrics = sum([Metrics([get_timestamps(count)]) for count in step_counts])
    metrics.set_index(date_range)
    metrics._df = metrics._df[metrics._df['step_count'] > 2]
    return metrics


if __name__ == "__main__":
    print(generate_metrics(days=10))