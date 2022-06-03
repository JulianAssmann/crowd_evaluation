from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional, Any
import numpy as np

from datasets import Dataset


class Evaluator(ABC):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    @abstractmethod
    def evaluate_worker(self, worker: int, *args, **kwargs) -> float:
        """Evaluates the given worker and returns the estimated error rate.

        :param worker: The worker to be evaluated
        :return: The estimated error rate for the given worker.
        """
        raise NotImplementedError()

    def evaluate_workers(self, workers: Union[List[int], np.ndarray], *args, **kwargs) -> np.ndarray:
        """Evaluates multiple workers at once given a desired confidence level.

        :param workers: The workers to be evaluated.
        :return: The estimated error rates for the given worker.
        """
        err_ests = np.zeros(len(workers), dtype=np.float32)
        for i, worker in enumerate(workers):
            err_est = self.evaluate_worker(worker, *args, **kwargs)
            err_ests[i] = err_est
        return err_ests
