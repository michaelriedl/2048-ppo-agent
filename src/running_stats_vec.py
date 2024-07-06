import numpy as np


class RunningStatsVec(object):
    """
    Running mean and variance computation for vectors.
    """

    def __init__(self):
        self.num_samples = np.zeros((1, 1), dtype=np.int64)
        self._mean = np.zeros((1, 1), dtype=np.float64)
        self._variance = np.zeros((1, 1), dtype=np.float64)

    def clear(self):
        """
        Clear the running statistics.

        This method resets the number of samples, mean, and variance to zero.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.num_samples = np.zeros((1, 1), dtype=np.int64)
        self._mean = np.zeros((1, 1), dtype=np.float64)
        self._variance = np.zeros((1, 1), dtype=np.float64)

    def push(self, x: np.ndarray):
        """
        Update the running statistics with new data.

        This method updates the running statistics with new data. The input data
        should have 2 dimensions, where the first dimension corresponds to the
        number of features and the second dimension corresponds to the number of
        samples.

        Parameters
        ----------
        x : np.ndarray
            Input data with shape (num_features, num_samples).

        Returns
        -------
        None
        """
        # Check if there are 2 dimensions
        if len(x.shape) != 2:
            raise ValueError("Input array should have 2 dimensions.")
        self._update_params(x)

    def _update_params(self, x: np.ndarray):
        # Check if the number of samples is greater than the current number of samples
        if x.shape[0] > self.num_samples.shape[0]:
            self.num_samples = np.append(
                self.num_samples,
                np.zeros((x.shape[0] - self.num_samples.shape[0], 1), dtype=np.int64),
                axis=0,
            )
            self._mean = np.append(
                self._mean,
                np.zeros((x.shape[0] - self._mean.shape[0], 1), dtype=np.float64),
                axis=0,
            )
            self._variance = np.append(
                self._variance,
                np.zeros((x.shape[0] - self._variance.shape[0], 1), dtype=np.float64),
                axis=0,
            )
        # Calculate the intermediate values needed for update
        sum_ns = self.num_samples[: x.shape[0]] + x.shape[1]
        prod_ns = self.num_samples[: x.shape[0]] * x.shape[1]
        delta2 = (x.mean(axis=1, keepdims=True) - self._mean[: x.shape[0]]) ** 2.0
        # Update the parameters
        self._mean[: x.shape[0]] = (
            self._mean[: x.shape[0]] * self.num_samples[: x.shape[0]]
            + x.mean(axis=1, keepdims=True) * x.shape[1]
        ) / sum_ns
        self._variance[: x.shape[0]] = (
            x.var(axis=1, keepdims=True) * x.shape[1]
            + self.num_samples[: x.shape[0]] * self._variance[: x.shape[0]]
            + delta2 * prod_ns / sum_ns
        ) / sum_ns
        self.num_samples[: x.shape[0]] = sum_ns

    @property
    def mean(self):
        return self._mean if self.num_samples.sum() else 0.0

    @property
    def variance(self):
        return self._variance if self.num_samples.sum() else 0.0

    @property
    def std(self):
        return np.sqrt(self._variance) if self.num_samples.sum() else 0.0
