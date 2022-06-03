import numpy as np
import pandas as pd

from crowd_evaluation import Evaluator
from datasets import Dataset


class MajorityEvaluator(Evaluator):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

    def evaluate_worker(self, worker: int, **kwargs) -> float:
        worker_samples = self.dataset.get_samples_for_worker(worker)
        majority_answers = self.dataset.get_majority_vote_for_samples(worker_samples)
        worker_answers = self.dataset.get_answers(worker, worker_samples)

        err_est = np.count_nonzero(np.where(majority_answers == worker_answers, 0, 1)) / len(worker_answers)
        return err_est

