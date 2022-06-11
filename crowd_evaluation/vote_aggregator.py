from datasets import Dataset
from typing import List, Union, Tuple
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
        "normal" classifies workers with an estimated error rate above the threshold as spammers.
        "conservative" classifies workers with the lower bound of the confidence interval above the threshold as spammers.
        "aggressive" classifies workers with the upper bound of the confidence interval above the threshold as spammers.
        :return: An array of the workers identified as spammers.
        """
        assert (p_ests is not None)
        if method == "normal":
            spam_filter = np.where(p_ests > threshold)[0]
            return workers[spam_filter]
        elif method == "conservative":
            if confs is None:
                raise ValueError("The confidence intervals cannot be None for method lower_bound")

            spam_filter = np.where(p_ests - confs > threshold)[0]
            return workers[spam_filter]
        elif method == "aggressive":
            if confs is None:
                raise ValueError("The confidence intervals cannot be None for method aggressive")
            pass

            spam_filter = np.where(p_ests + confs > threshold)[0]
            return workers[spam_filter]
        else:
            raise ValueError("Invalid method.")

    @staticmethod
    def majority_vote(dataset: Dataset,
                      samples: Union[List, np.ndarray],
                      blocked_workers: np.ndarray = None):

        labels = np.zeros(len(samples))
        for i, sample in enumerate(samples):
            # Get all the workers that worked on this sample
            workers = dataset.get_workers_for_sample(sample)

            if blocked_workers is not None:
                workers = np.setdiff1d(workers, blocked_workers)

            # Get the answers from all the workers that worked on this sample
            answers = dataset.get_answers_for_sample(workers, sample)
            answers = np.fromiter(answers.values(), dtype=int)

            labels[i] = (np.sum(answers) / len(answers))
        return labels > 0.5

    @staticmethod
    def weighted_vote(dataset: Dataset,
                      samples: Union[List, np.ndarray],
                      p_ests: dict,
                      selectivity: float = 0.5,
                      blocked_workers: np.ndarray = None) -> np.ndarray:
        confidence_intervals = dict(zip(dataset.workers, [0] * len(dataset.workers)))
        results, accuracies, overall_accuracy, min_overall_accuracy = VoteAggregator.weighted_vote_with_accuracies(
            dataset=dataset,
            samples=samples,
            p_ests=p_ests,
            confidence_intervals=confidence_intervals,
            confidence_level=0,
            selectivity=selectivity,
            blocked_workers=blocked_workers)
        return results

    @staticmethod
    def majority_vote_with_accuracies(dataset: Dataset,
                                      samples: Union[List, np.ndarray],
                                      p_ests: dict,
                                      confidence_intervals: dict,
                                      confidence_level: float,
                                      selectivity: float = 0.5,
                                      blocked_workers: np.ndarray = None) \
            -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Applies the majority vote to produce labels for the given samples and calculates the estimated label accuracies,
        the estimated overall accuracy and the guaranteed (smallest possible) accuracy.

        :param dataset: The dataset.
        :param samples: The samples for which the majority vote is to be calculated.
        :param p_ests: A dictionary with workers as keys and their estimated error rates as values.
        :param confidence_intervals: A dictionary with workers as keys and the confidence intervals for the
        estimated error rates as values.
        :param confidence_level: The confidence level with which the confidence interva ls of the workers were
        calculated.
        :param selectivity: The prior/probability of the result of a sample being 1.
        :param blocked_workers: A list of blocked workers. The votes of these workers are not taken into account.
        Defaults to None, meaning that no workers are blocked.
        :return: A tuple containing
            - the estimated labels for the given samples
            - the estimated accuracies for the labels
            - the estimated overall accuracy
            - the guaranteed accuracy (lower bound)
        """

        def beta_closure(answers: np.ndarray, error_rates: np.ndarray):
            return np.sum(answers) / len(answers)

        betas, ps_negs, ps_negs_worst, ps_pos, ps_pos_worst = VoteAggregator._calculate_labels_and_accuracies(
            blocked_workers, confidence_intervals, dataset, p_ests, samples, selectivity, beta_closure)

        # Calculate expected accuracy
        results = np.where(betas > 0, 1, 0)
        accuracies, min_overall_accuracy, overall_accuracy = VoteAggregator._calculate_overall_accuracies(
            confidence_level, ps_negs, ps_negs_worst, ps_pos, ps_pos_worst, results)

        return results, accuracies, overall_accuracy, min_overall_accuracy

    @staticmethod
    def weighted_vote_with_accuracies(dataset: Dataset,
                                      samples: Union[List, np.ndarray],
                                      p_ests: dict,
                                      confidence_intervals: dict,
                                      confidence_level: float,
                                      selectivity: float = 0.5,
                                      blocked_workers: np.ndarray = None) \
            -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Applies the weighted vote (as presented in Lemma 4 and Lemma 5 in "Evaluating the Crowd with Confidence")
        to produce labels for the given samples and calculates the estimated label accuracies,
        the estimated overall accuracy and the guaranteed (smallest possible) accuracy.

        :param dataset: The dataset.
        :param samples: The samples for which the majority vote is to be calculated.
        :param p_ests: A dictionary with workers as keys and their estimated error rates as values.
        :param confidence_intervals: A dictionary with workers as keys and the confidence intervals for the
        estimated error rates as values.
        :param confidence_level: The confidence level with which the confidence interva ls of the workers were
        calculated.
        :param selectivity: The prior/probability of the result of a sample being 1.
        :param blocked_workers: A list of blocked workers. The votes of these workers are not taken into account.
        Defaults to None, meaning that no workers are blocked.
        :return: A tuple containing
            - the estimated labels for the given samples
            - the estimated accuracies for the labels
            - the estimated overall accuracy
            - the guaranteed accuracy (lower bound)
        """
        if selectivity is None:
            alpha = 0
        else:
            alpha = np.log(selectivity / (1 - selectivity))

        def beta_closure(answers: np.ndarray, error_rates: np.ndarray):
            return np.sum(answers * np.log((1 - error_rates) / error_rates))

        betas, ps_negs, ps_negs_worst, ps_pos, ps_pos_worst = VoteAggregator._calculate_labels_and_accuracies(
            blocked_workers, confidence_intervals, dataset, p_ests, samples, selectivity, beta_closure)

        results = np.where(alpha + betas > 0, 1, 0)

        accuracies, min_overall_accuracy, overall_accuracy = VoteAggregator._calculate_overall_accuracies(
            confidence_level, ps_negs, ps_negs_worst, ps_pos, ps_pos_worst, results)

        return results, accuracies, overall_accuracy, min_overall_accuracy

    @staticmethod
    def _calculate_probabilities(selectivity: np.ndarray, error_rates: np.ndarray, answers: np.ndarray):
        """
        Calculates the probabilities P_1 and P_-1 as defined in Lemma 4 in "Evaluating the crowd with confidence"
        :param selectivity: The prior probability of the label being 1 (as a 1-element numpy array)
        :param error_rates: The error rates of the workers.
        :param answers: The answers of the workers.
        :return: A tuple containing P_1 and P_-1.
        """
        # Calculate probabilities p1 and p2 according to Lemma 4 in "Evaluating the crowd with confidence"
        p_pos = np.prod(np.concatenate((np.array([selectivity]),
                                        np.power(error_rates, (1 - answers) / 2),
                                        np.power(1 - error_rates, (1 + answers) / 2))))
        p_neg = np.prod(np.concatenate((np.array([1 - selectivity]),
                                        np.power(error_rates, (1 + answers) / 2),
                                        np.power(1 - error_rates, (1 - answers) / 2))))

        return p_pos, p_neg

    @staticmethod
    def _calculate_overall_accuracies(confidence_level, ps_negs, ps_negs_worst, ps_pos, ps_pos_worst, results):
        accuracies = np.where(results > 0,
                              ps_pos / (ps_pos + ps_negs),
                              ps_negs / (ps_pos + ps_negs))
        overall_accuracy = np.average(accuracies)
        # Calculate worst case accuracy
        accuracies_worst = np.where(results > 0,
                                    ps_pos_worst / (ps_pos_worst + ps_negs_worst),
                                    ps_negs_worst / (ps_pos_worst + ps_negs_worst))
        overall_accuracy_worst = np.average(accuracies_worst)
        min_overall_accuracy = 1 - (1-confidence_level + confidence_level * (1 - overall_accuracy_worst))
        return accuracies, min_overall_accuracy, overall_accuracy

    @staticmethod
    def _calculate_labels_and_accuracies(blocked_workers, confidence_intervals, dataset, p_ests, samples, selectivity,
                                         beta_closure):
        betas = np.zeros(len(samples))
        ps_pos = np.zeros(len(samples), dtype=np.float)
        ps_negs = np.zeros(len(samples), dtype=np.float)
        ps_pos_worst = np.zeros(len(samples), dtype=np.float)
        ps_negs_worst = np.zeros(len(samples), dtype=np.float)
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
            confs = np.array([confidence_intervals[i] for i in answers.keys()])

            # Transform the answers from {0, 1} to {-1, 1}
            answers = {k: -1 if v == 0 else 1 for k, v in answers.items()}
            answers = np.fromiter(answers.values(), dtype=int)

            error_rate_zero_filter = np.where(error_rates == 0)[0]
            error_rates = np.delete(error_rates, error_rate_zero_filter)
            answers = np.delete(answers, error_rate_zero_filter)
            confs = np.delete(confs, error_rate_zero_filter)

            beta = beta_closure(answers, error_rates)
            betas[i] = beta

            # Calculate probabilities p1 and p2 according to Lemma 4 in "Evaluating the crowd with confidence"
            p_pos, p_neg = VoteAggregator._calculate_probabilities(selectivity, error_rates, answers)
            ps_pos[i] = p_pos
            ps_negs[i] = p_neg

            # Calculate worst case probabilities p1 and p2 according to Lemma 4 in
            # "Evaluating the crowd with confidence"
            p_pos_worst, p_neg_worst = VoteAggregator._calculate_probabilities(selectivity, error_rates + confs,
                                                                               answers)
            ps_pos_worst[i] = p_pos_worst
            ps_negs_worst[i] = p_neg_worst
        return betas, ps_negs, ps_negs_worst, ps_pos, ps_pos_worst
