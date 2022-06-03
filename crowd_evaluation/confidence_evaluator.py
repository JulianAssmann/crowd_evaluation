from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional, Any
import numpy as np

from crowd_evaluation import Evaluator
from datasets import Dataset


class ConfidenceEvaluator(Evaluator):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

    def evaluate_worker(self, worker: int, *args, **kwargs) -> float:
        p, conf = self.evaluate_worker_with_confidence(worker, *args, **kwargs)
        return p

    @abstractmethod
    def evaluate_worker_with_confidence(self,
                                        worker: int,
                                        confidence: float = 0.9,
                                        *args, **kwargs) -> Tuple[float, float]:
        """Evaluates the given worker.
        Calculates the confidence interval based on the given confidence level.

        :param worker: The worker to be evaluated.
        :param confidence: The desired confidence level (used to calculate the confidence interval).
        :return: The estimated error rate and the confidence interval's half size
        """
        raise NotImplementedError()

    def evaluate_workers_with_confidence(self,
                                         workers: Union[List[int], np.ndarray],
                                         confidence: float = 0.9,
                                         *args, **kwargs) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Evaluates multiple workers at once given a desired confidence level.

        :param workers: The workers to be evaluated.
        :param confidence: The desired confidence level (used to calculate the confidence intervals).
        :return: The estimated error rates and confidence interval half sizes for the given workers.
        """
        ps = np.zeros(len(workers), dtype=np.float32)
        confs = np.zeros(len(workers), dtype=np.float32)
        for i, worker in enumerate(workers):
            p, conf = self.evaluate_worker_with_confidence(worker, confidence, *args, **kwargs)
            ps[i] = p
            confs[i] = conf
        return ps, confs

    @staticmethod
    def calculate_partials_f(q_ij: float, q_ik: float, q_jk: float) -> dict[int, Optional[Any]]:
        """Calculates the partial derivatives of function f according to Lemma 2.

        :param q_ij: The agreement rate between worker i and worker j
        :param q_ik: The agreement rate between worker i and worker k
        :param q_jk: The agreement rate between worker j and worker k
        :return: The partial derivatives in a dict with value
            - 1 representing df/dq_ij
            - 2 representing df/dq_ik
            - 3 representing df/dq_jk
        """
        dq_12 = -np.sqrt((q_ik - 0.5) / (8 * (q_ij - 0.5) * (q_jk - 0.5)))
        dq_13 = -np.sqrt((q_ij - 0.5) / (8 * (q_ik - 0.5) * (q_jk - 0.5)))
        dq_23 = np.sqrt((q_ij - 0.5) * (q_ik - 0.5) / (8 * (q_jk - 0.5) ** 3))

        partials = {
            1: dq_12,
            2: dq_13,
            3: dq_23
        }
        return partials

    @staticmethod
    def estimate_error_rate(q_ij, q_ik, q_jk) -> float:
        """Calculates the error rate estimate for worker i.

        :param q_ij: The agreement rate between worker i and worker j
        :param q_ik: The agreement rate between worker i and worker k
        :param q_jk: The agreement rate between worker j and worker k
        :return: The error rate estimate
        """
        q_ij, q_ik, q_jk = max(0.501, q_ij), max(0.501, q_ik), max(0.501, q_jk)
        return 0.5 - 0.5 * np.sqrt(
            (2 * q_ij - 1) * (2 * q_ik - 1) /
            (2 * q_jk - 1)
        )
