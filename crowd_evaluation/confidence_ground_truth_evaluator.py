from typing import Tuple

import numpy as np

from crowd_evaluation import ConfidenceEvaluator
from datasets import GroundTruthDataset
from scipy import stats


class ConfidenceGroundTruthEvaluator(ConfidenceEvaluator):
    def __init__(self, dataset: GroundTruthDataset):
        super().__init__(dataset)
        self.dataset = dataset

    def evaluate_worker(self, worker: int, *args, **kwargs) -> float:
        p_est, conf = self.evaluate_worker_with_confidence(worker)
        return p_est

    def evaluate_worker_with_confidence(self, worker: int,
                                        confidence: float = 0.9,
                                        wilson: bool = False,
                                        *args, **kwargs) -> Tuple[float, float]:
        print('Value of wilson', wilson)
        samples = self.dataset.get_samples_for_worker(worker)
        n = len(samples)

        truth = self.dataset.get_ground_truth_for_samples(samples)
        answers = self.dataset.get_answers(worker, samples)

        X = np.count_nonzero(np.where(truth == answers, 1, 0))
        p_est = 1 - X/n

        # Confidence calculation
        t = (1 + confidence) / 2
        z_t = stats.norm.ppf(t)

        if wilson:
            print('Evaluating using wilson score interval')
            nominator = z_t * np.sqrt(p_est * (1-p_est) / n + z_t**2 / (4 * n**2))
            denominator = 1 + 1/n * z_t**2
            conf = nominator / denominator
        else:
            print('Evaluating using std interval')
            conf = z_t * np.sqrt((p_est * (1 - p_est)) / n)

        return p_est, conf
