from datasets import Dataset
from typing import List, Union
import numpy as np


class VoteAggregator:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def majority_vote(self, samples: Union[List, np.ndarray] = None):
        if samples is None:
            samples = self.dataset.samples
        return self.dataset.get_majority_vote_for_samples(samples)

    def weighted_majority_vote(self,
                               samples: Union[List, np.ndarray],
                               dataset: Dataset,
                               p_ests: dict):
        pass

    def weighted_vote(self,
                      samples: Union[List, np.ndarray],
                      dataset: Dataset,
                      p_ests: dict,
                      selectivity: float = 0.5,
                      blocked_workers: np.ndarray = None):
        """
        As presented in Lemma 4 in "Evaluating the Crowd with Confidence"

        :param dataset: The dataset.
        :param samples: The samples we want the most likely estimated true labels for.
        :param p_ests: A dictionary of the estimated error rates with the worker as key and their corresponding error
        rates as values.
        :param selectivity: The prior/probability of a sample being positive (having 1 as the true value).
        :param blocked_workers: A list of blocked workers (i.e. spammers) that should not be considered for voting.
        """
        if samples is None:
            samples = self.dataset.samples

        alpha = np.log(selectivity / (1 - selectivity))
        betas = np.zeros(len(samples))
        for i, sample in enumerate(samples):
            # Get all the workers that worked on this sample
            workers = dataset.get_workers_for_sample(sample)

            if blocked_workers is not None:
                workers = np.setdiff1d(workers, blocked_workers)

            # Get the answers from all the workers that worked on this sample
            answers = dataset.get_answers_for_sample(workers, sample)
            # print('Workers involved in sample', sample, answers.keys())

            # Generate two arrays with the workers error rates and the answers respectively
            error_rates = np.array([p_ests[i] for i in answers.keys()])
            # print('Error rates', error_rates)

            # Transform the answers from {0, 1} to {-1, 1}
            answers = {k: -1 if v == 0 else 1 for k, v in answers.items()}
            answers = np.fromiter(answers.values(), dtype=int)
            # print('Answers', answers)

            beta = np.sum(answers * np.log((1 - error_rates) / error_rates))
            betas[i] = beta

        betas = np.array(betas)
        return np.where(alpha + betas > 0, 1, 0)
