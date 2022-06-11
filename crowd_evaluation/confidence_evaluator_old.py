from typing import List, Optional, Tuple, Union
import numpy as np
from numpy.random import default_rng
from scipy import stats

from . import ConfidenceEvaluator
from .utils import generate_superworkers_exhaustively, majority_vote
from datasets import Dataset


class ConfidenceEvaluatorOld(ConfidenceEvaluator):
    def __init__(self, dataset: Dataset, debug: bool = False):
        super().__init__(dataset)
        self.debug = debug

    def evaluate_workers_with_confidence(self,
                                         workers: Union[List[int], np.ndarray],
                                         confidence: float = 0.9,
                                         peer_count: Optional[int] = None,
                                         min_samples: Optional[int] = None,
                                         method: str = "exhaustive",
                                         wilson: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """

        :param workers:
        :param confidence:
        :param peer_count:
        :param min_samples:
        :param method:
        :param wilson: Whether to use the wilson score interval.
        :return:
        """
        ps = np.zeros(len(workers), dtype=np.float32)
        confs = np.zeros(len(workers), dtype=np.float32)
        for i, worker in enumerate(workers):
            if self.debug:
                print('Evaluating worker', i)
            p, conf = self.evaluate_worker_with_confidence(worker=worker,
                                                           peer_count=peer_count,
                                                           min_samples=min_samples,
                                                           method=method,
                                                           confidence=confidence,
                                                           wilson=wilson)
            ps[i] = p
            confs[i] = conf
        return ps, confs

    def evaluate_worker_with_confidence(self,
                                        worker: int,
                                        peer_count: Optional[int] = None,
                                        min_samples: Optional[int] = None,
                                        method: str = "greedy",
                                        confidence: float = 0.9,
                                        wilson: bool = True) -> Tuple[float, float]:
        """

        :param worker: The worker to be evaluated.
        :param peer_count: The maximum number of peers we want to find. None means that min_samples is the
        only factor influencing the number of peers.
        :param min_samples: The minimum number of samples the workers used to create superworkers have to share with
        the worker being evaluated. None means that peer_count is the only factor influencing the number of peers.
        :param method: The method used to generate the superworkers. Possible values are 'exhaustive' and 'greedy'.
        :param confidence: The desired confidence level (used to calculate the confidence intervals).
        :param wilson: Whether to use the wilson score interval.
        :return: The estimated error rate and its confidence interval for the given worker.
        """
        if self.debug:
            print('Evaluating worker', worker, 'with the', method, 'method...')
        peers, samples = self.dataset.find_peers_for_worker(worker, peer_count=peer_count, min_samples=min_samples)
        if len(peers) == 0:
            return 0, 0

        if method == "exhaustive":
            # Generate all possible superworker combinations (exhaustively).
            superworker_groups = generate_superworkers_exhaustively(peers)

            # The error rate estimates for worker
            p_estimates = []

            # Confidence intervals for worker, superworker1 and superworker2
            workers_conf_intervals: List[Tuple[int]] = []

            # Confidence intervals for worker
            worker_conf_intervals = []

            # Evaluate each possible superworker combination.
            for i, superworker_group in enumerate(superworker_groups):
                superworker1 = superworker_group[0]
                superworker2 = superworker_group[1]

                p_est, conf_intervals = self.evaluate_worker_with_superworkers(worker, superworker1, superworker2,
                                                                               samples, confidence=confidence)

                p_estimates.append(p_est)
                workers_conf_intervals.append(conf_intervals)
                worker_conf_intervals.append(conf_intervals[0])

            worker_conf_intervals = np.array(worker_conf_intervals)
            min_err_idx = np.argmin(worker_conf_intervals)

            # return
            #   superworker_groups[min_err_idx],
            #   p_estimates[min_err_idx],
            #   workers_conf_intervals[min_err_idx],
            #   samples,
            #   peers

            return p_estimates[min_err_idx][0], worker_conf_intervals[min_err_idx]
        elif method == "pruning":
            # TODO: Use pruning for superworker search
            raise NotImplementedError("Please implement this abstract method")
        elif method == "greedy":
            # Choose two random peers as initial superworkers
            rand = default_rng()
            indices = rand.choice(len(peers), size=len(peers), replace=False)
            superworker1 = [peers[indices[0]]]
            superworker2 = [peers[indices[1]]]

            # Measure initial superworkers confidence intervals
            p_est, conf = self.evaluate_worker_with_superworkers(
                worker, superworker1, superworker2, samples, confidence=confidence)

            i = 2
            while i < len(peers):
                # Add random peer to a copy of the superworker with the larger confidence interval
                new_superworker1 = superworker1.copy()
                new_superworker2 = superworker2.copy()
                if conf[1] > conf[2]:
                    superworker1.append(indices[i])
                elif conf[2] > conf[1]:
                    superworker2.append(indices[i])

                # Evaluate new superworkers
                new_p_est, new_conf = self.evaluate_worker_with_superworkers(
                    worker, new_superworker1, new_superworker2, samples, confidence=confidence, wilson=wilson)

                # Check if new superworkers reduce confidence interval. If so, add them to the superworker.
                if new_conf[1] < conf[1] or new_conf[2] < conf[2]:
                    p_est = new_p_est
                    conf = new_conf
                    superworker1 = new_superworker1
                    superworker2 = new_superworker2

                i += 1

            superworker_group = (worker, superworker1, superworker2)
            # return superworker_group, p_est, conf, samples, peers
            return p_est[0], conf[0]

    def evaluate_worker_with_superworkers(self,
                                          worker: int,
                                          superworker1: List,
                                          superworker2: List,
                                          samples: Union[List, np.ndarray],
                                          confidence: float = 0.9,
                                          wilson: bool = True):
        """Evaluates a given worker with the given superworkers.

        :param worker: The worker to evaluate
        :param superworker1: The list of workers that make up the first superworker.
        :param superworker2: The list of workers that make up the second superworker.
        :param samples: The indices of the samples based upon which to evaluate the worker.
        :param confidence: The required confidence level.
        :param wilson: Whether to use the wilson score interval.

        Returns:
            The estimated error rate p_est and its confidence interval half size for the given worker.
        """
        if self.debug:
            print('Evaluate worker', worker, 'with superworkers', superworker1, 'and', superworker2, '...')
        superworker_1_votes = np.zeros((len(superworker1), len(samples)))
        superworker_2_votes = np.zeros((len(superworker2), len(samples)))

        for i, w in enumerate(superworker1):
            superworker_1_votes[i, :] = self.dataset.get_answers(w, samples)[:]
        for i, w in enumerate(superworker2):
            superworker_2_votes[i, :] = self.dataset.get_answers(w, samples)[:]

        superworker1_vote = majority_vote(superworker_1_votes)
        superworker2_vote = majority_vote(superworker_2_votes)
        worker_vote = self.dataset.get_answers(worker, samples)

        # Calculate the fraction of agreement between the worker and two superworkers
        n = len(samples)
        q_12 = max(0.501, np.count_nonzero(worker_vote == superworker1_vote) / n)
        q_23 = max(0.501, np.count_nonzero(superworker1_vote == superworker2_vote) / n)
        q_13 = max(0.501, np.count_nonzero(worker_vote == superworker2_vote) / n)

        p1 = self.estimate_error_rate(q_12, q_13, q_23)
        p2 = self.estimate_error_rate(q_12, q_23, q_13)
        p3 = self.estimate_error_rate(q_13, q_23, q_12)

        p_est = np.abs(np.array([p1, p2, p3]))

        # if self.debug:
        #     print('q_12, q_23, q_13:', [q_12, q_23, q_13])
        #     print('p_est:', p_est)

        confidence_intervals = self.calculate_confidence_interval(
            qs=np.array([q_12, q_23, q_13]),
            p_est=p_est,
            confidence=confidence,
            n=len(samples),
            wilson=wilson)

        return p_est, confidence_intervals

    def calculate_confidence_interval(self,
                                      qs: np.ndarray,
                                      p_est: np.ndarray,
                                      confidence: float, n: int,
                                      wilson: bool = True):
        """Calculates the confidence interval sizes for the given p_est.

        :param qs: The disagreement numbers for each of the workers [q_12, q_23, q_13].
        :param p_est: The estimated error rates for the workers [p1, p2, p3]
        :param confidence: The required confidence level
        :param n: The number of samples
        :param wilson: Whether to use the wilson score interval.

        :return: A list of the confidence intervals for each worker.
        """

        q_12, q_23, q_13 = qs[0], qs[1], qs[2]
        p_est = np.clip(p_est, a_max=0.49, a_min=None)

        # Confidence calculation
        t = (1 + confidence) / 2

        # Calculate z_t to give the t'th percentile of the normal distribution
        z_t = stats.norm.ppf(t)

        if wilson:
            nominator = z_t * np.sqrt(p_est * (1 - p_est) / n + z_t ** 2 / (4 * n ** 2))
            denominator = 1 + 1 / n * z_t ** 2
            q_eps = nominator / denominator
        else:
            q_eps = z_t * np.sqrt((p_est * (1 - p_est)) / n)

        eps_12, eps_23, eps_13 = q_eps[0], q_eps[1], q_eps[2]

        # if self.debug:
        #     print('q_eps:', q_eps)

        max1 = self.estimate_error_rate(q_12 + eps_12, q_13 + eps_13, q_23 - eps_23)
        min1 = self.estimate_error_rate(q_12 - eps_12, q_13 - eps_13, q_23 + eps_23)
        half_size1 = (max1 - min1) / 2

        max2 = self.estimate_error_rate(q_23 + eps_23, q_12 + eps_12, q_13 - eps_13)
        min2 = self.estimate_error_rate(q_23 - eps_23, q_12 - eps_12, q_13 + eps_13)
        half_size2 = (max2 - min2) / 2

        max3 = self.estimate_error_rate(q_13 + eps_13, q_23 + eps_23, q_12 - eps_12)
        min3 = self.estimate_error_rate(q_13 - eps_13, q_23 - eps_23, q_12 + eps_12)
        half_size3 = (max3 - min3) / 3

        half_sizes = np.array([half_size1, half_size2, half_size3])
        # if self.debug:
        #     print('half sizes:', half_sizes)

        return np.abs(half_sizes)
