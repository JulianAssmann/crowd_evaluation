from abc import abstractclassmethod
from typing import List, Union
from .dataset import Dataset
import numpy as np
import pandas as pd


class GroundTruthDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame,
                 prefilter_mode: str = None,
                 prefilter_threshold: float = 0.4,
                 debug: bool = False):
        """
            dataframe (pandas.DataFrame): The data of the dataset. Should include the follwing columns:
                - worker
                - sample
                - answer
                - truth
            true_error_rates: A dictionary with workers as keys and their true error rates as values.
        """
        super().__init__(dataframe, prefilter_mode, prefilter_threshold, debug)

        self._measured_error_rates = dict()
        for worker in self.workers:
            answers = self.df[self.df['worker'] == worker]['answer'].to_numpy()
            truths = self.df[self.df['worker'] == worker]['truth'].to_numpy()
            self._measured_error_rates[worker] = np.count_nonzero(answers != truths) / float(len(answers))

    def get_measured_error_rates_for_workers(self, workers: Union[List[int], np.ndarray]) -> np.ndarray:
        """Returns the measured error rates based on the gold standard for the given workers.

        Args:
            workers (Union[List[int], np.ndarray]): A list of workers.
        Returns:
            numpy.ndarray: A list of the measured error rates.
        """
        error_rates = []
        for key in workers:
            error_rates.append(self._measured_error_rates[key])
        return np.array(error_rates)

    def get_measured_error_rate_for_worker(self, worker: int) -> float:
        """Returns the measured error rate based on the gold standard for the given worker.

        Args:
            worker (int): A worker
        Returns:
            float: The measured error rate.
        """
        return self._measured_error_rates[worker]

    def get_ground_truth_for_samples(self, samples: np.ndarray):
        """Returns the ground truth for the given samples.

        Args:
            samples (np.ndarray): The samples we want the ground truth for

        Returns:
            np.ndarray: The ground truth for the given samples ordered by samples.
        """
        df_f = self.df.drop_duplicates(subset='sample')
        return df_f[df_f['sample'].isin(samples)].sort_values(by='sample')['truth'].to_numpy()

    def get_measured_error_rate_for_superworker(self, superworker: Union[List[int], np.ndarray], samples):
        """Returns the measured error rate for the given superworker based on the ground truth.

        Args:
            superworker (Union[List[int], np.ndarray]): The superworker we want the measured error rate for.
            samples: The samples to evaluate the measured error for.

        Returns:
            double: The measured error rate for the given superworker.
        """
        # TODO
        raise NotImplementedError("Please implement this abstract method")