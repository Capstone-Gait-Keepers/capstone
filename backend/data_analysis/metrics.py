import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import hilbert
from scipy.stats import entropy
from typing import List


class Metrics:
    """
    A collection of the following gait metrics:

    - Step Count: Total number of steps recorded.
    - Stride Time Gait Asymmetry (STGA): A measure of the difference in stride time between the left and right legs.
    - Stride Time: The average time between steps of the same foot.
    - Cadence: The average number of steps per second.
    - Stride Time Coefficient of Variation: A measure of the regularity of the stride time. Low values indicate high regularity, which is good.
    - Stride Time Phase Synchronization Index: A measure of the phase synchronization between the left and right legs. High values indicate high phase synchronization, which is good.
    - Stride Time Conditional Entropy: A measure of the regularity of the stride time. Low values indicate high regularity, which is good
    """

    summed_vars = set(['step_count'])

    STEP_REQUIREMENTS = {
        'step_count': 0,
        'STGA': 3,
        'stride_time': 3,
        'cadence': 2,
        'var_coef': 6,
        'phase_sync': 8,
        'conditional_entropy': 6,
    }

    stats = pd.DataFrame.from_dict({
            'STGA': [0.015,0.050,0.1,0.2,0.15], # TODO: FAKE NUMBERS
            'stride_time': [1.036,0.1,1.128,0.107,0.092],
            'cadence': [1.692,0.202,1.588,0.197,0.107],
            'var_coef': [0.017,0.05,0.031,0.012,0.14],
            'phase_sync': [0.812,0.001,0.788,0.045,np.nan],
            'conditional_entropy': [0.007,0.015,0.054,0.094,np.nan],
        },
        columns=['ctrl_mu', 'ctrl_std', 'bad_mu', 'bad_std', 'measure_std'],
        orient='index',
    )

    def __init__(self, timestamp_groups: List[np.ndarray], recording_id=0):
        func_map = self.get_func_map()
        self.keys = self.get_keys()
        if len(timestamp_groups) == 0:
            self._df = pd.DataFrame({key: [np.nan] for key in self.keys})
            self._df['step_count'] = [0]
        else:
            data = {key: [func_map[key](timestamps) for timestamps in timestamp_groups] for key in self.keys}
            self._df = pd.DataFrame.from_dict(data)
        self._df['recording_id'] = [recording_id] * len(self._df)

    @classmethod
    def get_func_map(cls):
        return {
            'step_count': len,
            'STGA': cls._get_STGA,
            'stride_time': cls._get_stride_time,
            'cadence': cls._get_cadence,
            'var_coef': cls._get_stride_time_CV,
            'phase_sync': cls._get_phase_sync,
            'conditional_entropy': cls._get_conditional_entropy,
        }

    @staticmethod
    def get_control() -> pd.Series:
        return Metrics.stats['ctrl_mu']

    @staticmethod
    def get_keys():
        return list(Metrics.STEP_REQUIREMENTS.keys())

    @staticmethod
    def get_metric_names():
        return list(Metrics.stats.index)

    def __getitem__(self, key):
        return self.aggregate_metric(self._df[key], weights=self._df['step_count'])

    @staticmethod
    def aggregate_metric(data: pd.Series, weights=None):
        if len(data) == 0 or data.isna().all() or (weights is not None and weights.isna().all()):
            return np.nan
        if data.name in Metrics.summed_vars:
            return data.sum()
        return Metrics.nanaverage(data, weights=weights)

    @staticmethod
    def nanaverage(a: np.ndarray, weights: np.ndarray):
        return np.nansum(a*weights)/((~np.isnan(a))*weights).sum()

    def __len__(self):
        return len(self._df)

    @property
    def recordings(self):
        return self._df['recording_id'].nunique()

    @staticmethod
    def _get_STGA(timestamps: np.ndarray):
        if len(timestamps) < Metrics.STEP_REQUIREMENTS['STGA']:
            return np.nan
        stride_times = np.diff(timestamps)
        # TODO: Does this match literature?
        # https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=aeee9316f2a72d0f89e59f3c5144bf69a695730b
        return np.abs(np.mean(stride_times[1:] / stride_times[:-1]) - 1) / np.mean(stride_times)

    @staticmethod
    def _get_cadence(timestamps: np.ndarray):
        if len(timestamps) < Metrics.STEP_REQUIREMENTS['cadence']:
            return np.nan
        return 1 / np.mean(np.diff(timestamps))

    @staticmethod
    def _get_stride_time(timestamps):
        if len(timestamps) < Metrics.STEP_REQUIREMENTS['stride_time']:
            return np.nan
        return np.mean(Metrics._get_stride_times(timestamps))

    @staticmethod
    def _get_stride_time_CV(timestamps: np.ndarray):
        if len(timestamps) < Metrics.STEP_REQUIREMENTS['var_coef']:
            return np.nan
        return Metrics._get_var_coef(Metrics._get_stride_times(timestamps))

    @staticmethod
    def _get_stride_times(timestamps: np.ndarray) -> np.ndarray:
        if len(timestamps) < 2:
            return np.empty((0))
        a, b = Metrics._get_side_stride_times(timestamps)
        stride_times = np.empty((a.size + b.size,), dtype=timestamps.dtype)
        stride_times[0::2] = a
        stride_times[1::2] = b
        return stride_times

    @staticmethod
    def _get_side_stride_times(timestamps: np.ndarray) -> np.ndarray:
        if len(timestamps) < 2:
            return np.empty((0))
        a = np.diff(timestamps[::2])
        b = np.diff(timestamps[1::2])
        return a, b  

    @staticmethod
    def _get_var_coef(dist):
        """General formula for coefficient of variation"""
        if len(dist) < 3:
            return np.nan
        return np.std(dist) / np.mean(dist)

    @staticmethod
    def _get_phase_sync(timestamps: np.ndarray, num_bins=8):
        if len(timestamps) < Metrics.STEP_REQUIREMENTS['phase_sync']:
            return np.nan
        if len(timestamps) % 2 != 0:
            timestamps = np.copy(timestamps)[:-1]
        left_stride_times, right_stride_times = Metrics._get_side_stride_times(timestamps)
        analytic_signal1 = hilbert(left_stride_times)
        analytic_signal2 = hilbert(right_stride_times)
        phase1 = (np.unwrap(np.angle(analytic_signal1)) / (2 * np.pi)) % 1
        phase2 = (np.unwrap(np.angle(analytic_signal2)) / (2 * np.pi)) % 1
        phase_difference = phase1 - phase2
        H = Metrics._calculate_shannon_entropy(phase_difference, num_bins)
        H_max = np.log2(num_bins)
        return (H_max - H) / H_max

    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7247739
    @staticmethod
    def _get_conditional_entropy(timestamps: np.ndarray, num_bins=8):
        if len(timestamps) < Metrics.STEP_REQUIREMENTS['conditional_entropy']:
            return np.nan
        if len(timestamps) % 2 != 0:
            timestamps = np.copy(timestamps)[:-1]
        left_stride_times, right_stride_times = Metrics._get_side_stride_times(timestamps)
        left_cond_entropy = Metrics._calculate_cond_entropy(left_stride_times, right_stride_times, num_bins)
        right_cond_entropy = Metrics._calculate_cond_entropy(right_stride_times, left_stride_times, num_bins)
        return (left_cond_entropy + right_cond_entropy) / 2

    @staticmethod
    def _calculate_shannon_entropy(values: np.ndarray, num_bins: int, bin_range=(-np.pi, np.pi)):
        if len(values) <= 1:
            raise ValueError('Stride times must have at least 2 elements.')
        bins = Metrics._get_hist_bins(values, num_bins, bin_range)
        counts, bins = np.histogram(values, bins=bins)
        probabilities = counts / sum(counts)
        return entropy(probabilities, base=2)

    @staticmethod
    def _calculate_cond_entropy(S1: np.ndarray, S2: np.ndarray, num_bins: int, bin_range=(0, 3)):
        """
        Calculate the conditional entropy H(S1|S2) given two 1D numpy arrays S1 and S2.
        """
        # Joint distribution of S1 and S2
        bins = Metrics._get_hist_bins(np.concatenate([S1, S2]), num_bins, bin_range)
        p_joint, _, _ = np.histogram2d(S1, S2, bins=bins, density=True)
        p_S2 = np.sum(p_joint, axis=0) # Marginal distribution of S2
        H_S1_S2 = entropy(p_joint.flatten(), base=2) # H(S1, S2)
        H_S2 = entropy(p_S2, base=2) # H(S2)
        H_S1_given_S2 = H_S1_S2 - H_S2 # H(S1 | S2)
        return H_S1_given_S2

    @staticmethod
    def _get_hist_bins(values, num_bins: int, bin_range: tuple):
        if num_bins > len(values):
            num_bins = len(values) - 1
        bin_range = (min(values.min(), bin_range[0]), max(values.max(), bin_range[1]))
        return np.linspace(*bin_range, num_bins)

    def __add__(self, other: 'Metrics'):
        """Combines two Metrics objects by averaging their values."""
        if not isinstance(other, Metrics):
            raise ValueError(f'Can only add Metrics to Metrics, not {type(other)}. ({other})')
        if not len(self):
            return other
        if not len(other):
            return self
        last_recording_id = self._df['recording_id'].max()
        other._df['recording_id'] += last_recording_id + 1
        self._df = pd.concat([self._df, other._df], ignore_index=True, sort=False)
        return self

    def __radd__(self, other: 'Metrics | int'):
        """Combines two Metrics objects by averaging their values."""
        return self if other == 0 else self.__add__(other)

    def error(self, truth: 'Metrics', absolute=False, normalize=False) -> pd.DataFrame:
        """Returns the % error between two Metrics objects. Groups by recording_id."""
        if not isinstance(truth, Metrics):
            raise ValueError('Can only compare Metrics to Metrics.')
        if truth.recordings != self.recordings:
            raise ValueError(f'Cannot compare Metrics of different lengths. truth.recordings ({len(truth.recordings)}) != self.recordings ({self.recordings}) (len(self) = {len(self)})')
        self_rec_metrics = self.by_tag()
        truth_rec_metrics = truth.by_tag()
        error = self_rec_metrics - truth_rec_metrics
        if absolute:
            error = error.abs()
        if normalize:
            for key in Metrics.get_keys():
                if key not in Metrics.summed_vars:
                    error[key] /= truth_rec_metrics[key]
        # Where they are both NaN, the error is 0
        error = error.where(error != np.inf, np.nan)
        error = error.where(truth_rec_metrics.notna() | self_rec_metrics.notna(), 0)
        # Add classification bool
        correct_class = (self_rec_metrics['step_count'] > 0) ^ (truth_rec_metrics['step_count'] == 0)
        correct_class.name = "correct_class"
        error = pd.concat([correct_class, error], axis=1)
        return error

    def by_tag(self, smooth_window=0) -> pd.DataFrame:
        df = self._df.groupby('recording_id').apply(self.aggregate)
        if smooth_window:
            df = df.rolling(smooth_window, min_periods=1).mean()
        return df

    def set_index(self, new_ids: list):
        old_ids = self._df['recording_id'].unique()
        if len(new_ids) != len(old_ids):
            raise ValueError(f'New IDs ({len(new_ids)}) must be the same length as old IDs ({len(old_ids)}).')
        self._df['recording_id'].replace(dict(zip(old_ids, new_ids)), inplace=True)

    @staticmethod
    def aggregate(data: pd.DataFrame):
        """
        Aggregates a group of metrics into a single row. All metrics are assumed to be
        from the same recording. The `summed_vars` are summed, while all other metrics
        are averaged, weighted by the step count.
        """
        if len(data) == 0:
            return pd.Series({key: np.nan for key in Metrics.get_keys()})
        results = {}
        for metric in Metrics.get_keys():
            results[metric] = Metrics.aggregate_metric(data[metric], weights=data['step_count'])
        return pd.Series(results)

    def __str__(self) -> str:
        return str(self._df)
    
    def plot(self, metrics=None):
        keys = metrics or self.get_metric_names()
        data = self.by_tag()[keys]
        fig = make_subplots(rows=len(keys), cols=1, shared_xaxes=True, subplot_titles=keys)
        for i, key in enumerate(keys):
            fig.add_trace(go.Scatter(x=data.index, y=data[key], mode='lines+markers', name=key), row=i+1, col=1)
        fig.update_layout(showlegend=False)
        fig.show()

    def significant_change(self) -> pd.Series:
        """Detects significant changes in a series, based on CUSUM SPC"""
        d = {}
        w = Metrics.stats['ctrl_std']
        for metric in self.get_metric_names():
            pos_change = self._positive_change(self._df[metric], w[metric])
            neg_change = self._positive_change(-self._df[metric], w[metric])
            if pos_change != -1:
                d[metric] = pos_change
            elif neg_change != -1:
                d[metric] = neg_change
            else:
                d[metric] = -1
        return pd.Series(d) > 0

    @staticmethod
    def _positive_change(metric: pd.Series, w: float) -> int:
        """
        Detects significant changes in a series, based on CUSUM SPC. W is the threshold for change.
        Commonly, w = 1/2 * | u1 - u2 |, where u1 and u2 are the means of the control and bad groups.
        Returns the index of the first significant change, or -1 if no change is detected.
        """
        Sn = 0
        for i in range(1, len(metric)):
            Snext = max(0, Sn + metric.iloc[i] - metric.iloc[i - 1] - w)
            if Snext > 0:
                return i
            Sn = Snext
        return -1


if __name__ == "__main__":
    from generate_dummies import generate_metrics, decay
    days = 30
    asymmetry = decay(days, 0, 0.2)
    cadence = decay(days, 2, 2)
    var = decay(days, 0.02, 0.02)
    data = generate_metrics(days=days, cadence=cadence, asymmetry=asymmetry, var=var)
    print(data.significant_change())
    data.plot(['stride_time', 'var_coef', 'STGA'])
