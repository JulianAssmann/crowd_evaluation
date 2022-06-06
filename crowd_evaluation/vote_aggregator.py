from datasets import Dataset
from typing import List, Union
import numpy as np


class VoteAggregator:
    @staticmethod
    def filter_spammers(
            workers: np.ndarray,
            threshold: float,
            p_ests: np.ndarray,
            confs: np.ndarray = None,
            method="error rates",

    ) -> np.ndarray:
        """
        Identifies spammers amongst the given workers by their error rate and threshold and returns an array of the
        workers identified as spammers.

        :param workers: The workers to be evaluated.
        :param threshold: The threshold for identifying a worker as spammer.
        :param p_ests: The error rate estimations for the given workers.
        :param confs: The half-sizes of the confidence intervals for the given workers.
        Must not be None for methods "lower bound" and "upper bound".
        :param method: The method by which to classify spammers.
        "error rates" classifies workers with an estimated error rate above the threshold as spammers.
        "lower bound" classifies workers with the lower bound of the confidence interval above the threshold as spammers.
        "upper bound" classifies workers with the upper bound of the confidence interval above the threshold as spammers.
        :return: An array of the workers identified as spammers.
        """
        assert(p_ests is not None)
        if method == "error rates":
            spam_filter = np.where(p_ests > threshold)[0]
            return workers[spam_filter]
        elif method == "lower bound":
            if confs is None:
                raise ValueError("The confidence intervals cannot be None for method lower_bound")

            spam_filter = np.where(p_ests - confs > threshold)[0]
            return workers[spam_filter]
        elif method == "upper bound":
            if confs is None:
                raise ValueError("The confidence intervals cannot be None for method lower_bound")
            pass

            spam_filter = np.where(p_ests + confs > threshold)[0]
            return workers[spam_filter]
        else:
            raise ValueError("Invalid method.")

    @staticmethod
    def majority_vote(dataset: Dataset, samples: Union[List, np.ndarray] = None):
        if samples is None:
            samples = dataset.samples
        return dataset.get_majority_vote_for_samples(samples)

    @staticmethod
    def weighted_vote(dataset: Dataset,
                      samples: Union[List, np.ndarray],
                      p_ests: dict,
                      selectivity: float,
                      blocked_workers: np.ndarray):
        """
        As presented in Lemma 4 in "Evaluating the Crowd with Confidence"

        :param dataset: The dataset.
        :param samples: The samples we want the most likely estimated true labels for.
        :param p_ests: A dictionary of the estimated error rates with the worker as key and their corresponding error
        rates as values.
        :param selectivity: The prior/probability of a sample being positive (having 1 as the true value).
        :param blocked_workers: A list of blocked workers (i.e. spammers) that should not be considered for voting.
        """
        if selectivity is None:
            alpha = 0
        else:
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

            error_rate_zero_filter = np.where(error_rates == 0)[0]
            error_rates = np.delete(error_rates, error_rate_zero_filter)
            answers = np.delete(answers, error_rate_zero_filter)

            beta = np.sum(answers * np.log((1 - error_rates) / error_rates))
            betas[i] = beta



        betas = np.array(betas)
        results = np.where(alpha + betas > 0, 1, 0)

        return results
