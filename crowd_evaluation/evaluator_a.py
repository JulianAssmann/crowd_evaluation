import numpy as np
from scipy import stats
from typing import Tuple

from datasets import Dataset
from crowd_evaluation.confidence_evaluator import ConfidenceEvaluator


# Implements case A (3-worker binary regular) from the paper.
# Only supports binary classes.
class ConfidenceEvaluatorA(ConfidenceEvaluator):
    def __init__(self, dataset: Dataset, debug: bool = False):
        super().__init__(dataset)
        self.debug = debug

    def evaluate_worker_with_confidence(self,
                                        worker: int,
                                        confidence: float = 0.9,
                                        method: str = 'std') -> Tuple[float, float]:
        workers = set(self.dataset.workers)
        workers.discard(worker)
        worker1 = workers.pop()
        worker2 = workers.pop()

        q_12 = max(0.501, self.dataset.fraction_of_agreement_between(worker, worker1)[0])
        q_13 = max(0.501, self.dataset.fraction_of_agreement_between(worker, worker2)[0])
        q_23 = max(0.501, self.dataset.fraction_of_agreement_between(worker1, worker2)[0])

        p1 = self.estimate_error_rate(q_12, q_13, q_23)
        p2 = self.estimate_error_rate(q_12, q_23, q_13)
        p3 = self.estimate_error_rate(q_13, q_23, q_12)

        conf_interval = self.calc_conf_interval(p1, p2, p3, q_12, q_13, q_23, confidence=confidence)

        return np.abs(p1), conf_interval

    def calc_conf_interval(self, p1: float, p2: float, p3: float, q_12: float, q_13: float, q_23: float,
                           confidence: float):
        """
        Calculates the confidence interval for worker 1.
        :param p1: The estimated error rate for worker 1.
        :param p2: The estimated error rate for worker 2.
        :param p3: The estimated error rate for worker 3.
        :param q_12: The rate of agreement between worker 1 and worker 2.
        :param q_23: The rate of agreement between worker 2 and worker 3.
        :param q_13: The rate of agreement between worker 1 and worker 3.
        :param confidence: The desired confidence.
        :return: The half size of the confidence interval.
        """
        partials = self.calculate_partials_f(q_12, q_13, q_23)
        covariances = self.calc_covariances(p1, p2, p3, q_12, q_13, q_23)

        dev = 0
        for i in range(1, 4):
            for j in range(1, 4):
                dev += partials[i] * partials[j] * covariances[(i, j)]
        dev = np.sqrt(dev)

        # We want an interval with confidence c
        t = (1 + confidence) / 2
        # Calculate z_t to give the t'th percentile of the normal distribution
        z_t = stats.norm.ppf(t)
        confidence_interval = z_t * dev

        return confidence_interval

    def calc_covariances(self, p1: float, p2: float, p3: float, q_12: float, q_13: float, q_23: float) \
            -> dict[tuple[int, int], float]:
        """Calculates the covariances according to Lemma 1.

        :param p1: The estimated error rate of worker 1.
        :param p2: The estimated error rate of worker 2.
        :param p3: The estimated error rate of worker 3.
        :param q_12: The rate of agreement between worker 1 and worker 2.
        :param q_13: The rate of agreement between worker 1 and worker 3.
        :param q_23: The rate of agreement between worker 2 and worker 3.
        :return: A dictionary containing the covariances where the key of the dictionary is a tuple describing the
        covariant of which variables this is (1: Q_12, 2: Q_13, 3: Q_23).
        """
        # X_1 := Q_12
        # X_2 := Q_13
        # X_3 := Q_23
        n = self.dataset.num_samples

        # cov_11 = Cov(Q_12, Q_12)
        cov_11 = q_12 * (1 - q_12) / n
        # cov_22 = Cov(Q_13, Q_13)
        cov_22 = q_13 * (1 - q_13) / n
        # cov_33 = Cov(Q_13, Q_13)
        cov_33 = q_23 * (1 - q_23) / n

        # cov_12 = Cov(Q_12, Q_13)
        cov_12 = (p1 * (1 - p1) * (2 * q_23 - 1)) / n
        # cov_13 = Cov(Q_12, Q_23)
        cov_13 = (p2 * (1 - p2) * (2 * q_13 - 1)) / n
        # cov_23 = Cov(Q_13, Q_23)
        cov_23 = (p3 * (1 - p3) * (2 * q_12 - 1)) / n

        covariances = {
            (1, 1): cov_11,
            (2, 2): cov_22,
            (3, 3): cov_33,
            (1, 2): cov_12,
            (2, 1): cov_12,
            (1, 3): cov_13,
            (3, 1): cov_13,
            (2, 3): cov_23,
            (3, 2): cov_23,
        }
        return covariances
