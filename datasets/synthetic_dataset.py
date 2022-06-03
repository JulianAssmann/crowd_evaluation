from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from .ground_truth_dataset import GroundTruthDataset


class SyntheticDataset(GroundTruthDataset):

    def __init__(self,
                 num_samples: int = 1000,
                 num_workers: int = 3,
                 max_error_perc: float = 10.0,
                 p_true: Optional[List[float]] = None,
                 sample_fractions: Optional[List[float]] = None):
        """
        :param num_samples: The total number of samples to be generated.
        :param num_workers: The number of workers to be generated.
        :param max_error_perc: The maximum error rate a worker can have. Only applies when p_true is None.
        :param p_true: The true error rates for the workers. Defaults to None, meaning they will be generated randomly.
        :param sample_fractions: The fraction of total samples each worker annotates. Defaults to None, meaning each
        worker will annotate each sample.
        """
        self._workers = np.arange(0, num_workers, dtype=np.int64)
        self._samples = np.arange(0, num_samples, dtype=np.int64)

        if sample_fractions is None:
            self._sample_fractions = [1] * num_workers
        else:
            assert (len(sample_fractions) == num_workers)
            self._sample_fractions = sample_fractions

        if p_true is None:
            self._p_true = np.random.random(size=self.num_workers) / (100 / max_error_perc)
        else:
            assert (len(p_true) == num_workers)
            self._p_true = np.array(p_true)

        df = self._simulate_voting()
        super().__init__(df)

    def get_true_error_rates_for_workers(self, workers: Union[List[int], np.ndarray]) -> np.ndarray:
        """Returns the true error rates based on the gold standard for the given workers.

        Args:
            workers (Union[List[int], np.ndarray]): A list of workers.
        Returns:
            numpy.ndarray: A list of the true error rates.
        """
        return self._p_true[workers]

    def get_measured_error_rate_for_worker(self, worker: int) -> float:
        """Returns the true error rate based on the gold standard for the given worker.

        Args:
            worker (int): A worker
        Returns:
            float: The true error rate.
        """
        return self._p_true[worker]

    def _simulate_voting(self) -> pd.DataFrame:
        """Simulates the process of workers voting on binary tasks with synthetic data.
        """
        num_samples = len(self.samples)
        # Generate true labels
        y_true = np.random.randint(0, high=2, size=num_samples)

        data_workers = np.array([], dtype=np.int64)
        data_samples = np.array([], dtype=np.int64)
        data_answers = np.array([], dtype=np.int0)
        data_truth = np.array([], dtype=np.int0)
        data_error = np.array([], dtype=np.int0)

        Xs = []
        for i in range(self.num_workers):
            num_samples_for_worker = int(self._sample_fractions[i] * num_samples)

            # The samples annotated by this worker
            samples = np.sort(np.random.choice(self.samples, size=num_samples_for_worker, replace=False))

            # The ground truth for the samples annotated by this worker
            truth = y_true[samples]

            # Invert y_true in order to get wrong answers
            truth_inv = 1 - truth

            # Deviate with probability p_i from y_true (0=wrong answer, 1=right answer)
            X = np.random.choice([0, 1], size=num_samples_for_worker, p=[self._p_true[i], 1 - self._p_true[i]])
            Xs.append(X)

            # With proabiblity p_i choose the wrong answer
            answers = np.where(X == 1, truth, truth_inv)

            data_workers = np.append(data_workers, np.repeat(i, num_samples_for_worker))
            data_samples = np.append(data_samples, samples)
            data_answers = np.append(data_answers, answers)
            data_truth = np.append(data_truth, truth)
            data_error = np.append(data_error, 1 - X)

        data = {
            'worker': data_workers,
            'sample': data_samples,
            'answer': data_answers,
            'truth': data_truth,
            'error': data_error
        }

        return pd.DataFrame(data)

    def get_measured_error_rate_for_superworker(self, superworker: Union[List[int], np.ndarray], samples):
        raise NotImplementedError()
