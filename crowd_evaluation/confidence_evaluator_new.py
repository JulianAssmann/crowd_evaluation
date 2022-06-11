import numpy as np
from numpy.linalg import LinAlgError
from scipy import stats
from typing import Tuple, Union, List

from datasets import Dataset
from crowd_evaluation.confidence_evaluator import ConfidenceEvaluator


# Implements case C (m-worker binary non-regular) from the paper.
# Only supports binary classes.
class ConfidenceEvaluatorNew(ConfidenceEvaluator):
    def __init__(self, dataset: Dataset, debug: bool = False):
        super().__init__(dataset)
        self.debug = debug

    def evaluate_workers_with_confidence(self, workers: Union[List[int], np.ndarray],
                                         confidence: float = 0.9,
                                         method: str = 'std',
                                         min_shared_samples: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluates multiple workers at once given a desired confidence level.

        :param method: The method used to estimate the confidence interval. std or wilson.
        :param workers: The workers to be evaluated.
        :param confidence: The desired confidence level (used to calculate the confidence intervals).
        :param min_shared_samples: The minimum number of shared samples for triplets of workers to be considered in the
        calculations.
        :return:
        """
        ps = np.zeros(len(workers), dtype=np.float32)
        confs = np.zeros(len(workers), dtype=np.float32)
        for i, worker in enumerate(workers):
            p, conf = self.evaluate_worker_with_confidence(worker=worker,
                                                           confidence=confidence,
                                                           method=method,
                                                           min_shared_samples=min_shared_samples)
            ps[i] = p
            confs[i] = conf
            if self.debug:
                print('\n')
        return ps, confs

    def evaluate_worker_with_confidence(self,
                                        worker: int,
                                        confidence: float = 0.9,
                                        min_shared_samples: int = 1,
                                        method: str = 'std') -> Tuple[float, float]:
        if self.debug:
            print('Evaluating worker', worker)

        # Step 1: Form pairs
        pairs = self.generate_peer_pairs_for_worker(worker, min_shared_samples)
        l = len(pairs)

        if l == 0:
            if self.debug:
                print("There is no worker pair that shares at least " + str(min_shared_samples)
                      + " samples with the given worker " + str(worker))
            return np.nan, np.nan

        # Step 2: Apply 3-worker method per triple
        p_ki = dict()
        dev_ki = dict()
        d_kij = dict()
        for i in range(l):
            d_kij = dict()
            for j in self.dataset.workers:
                d_kij[j] = dict()
        # d_kij = np.zeros((l, m, m))

        for k in range(l):
            w_j1, w_j2 = pairs[k]

            if self.debug:
                print('Evaluating pair', (k + 1), 'out of', l, ': {', w_j1, ',', w_j2, '}',
                      ', no_shared_samples:', self.dataset.num_shared_samples_for_workers(worker, w_j1, w_j2))

            q_12 = max(0.501, self.dataset.fraction_of_agreement_between(worker, w_j1)[0])
            q_13 = max(0.501, self.dataset.fraction_of_agreement_between(worker, w_j2)[0])
            q_23 = max(0.501, self.dataset.fraction_of_agreement_between(w_j1, w_j2)[0])

            p1 = self.estimate_error_rate(q_12, q_13, q_23)
            p2 = self.estimate_error_rate(q_12, q_23, q_13)
            p3 = self.estimate_error_rate(q_13, q_23, q_12)

            partials = self.calculate_partials_f(q_12, q_13, q_23)
            covariances = self.calc_covariances(worker, w_j1, w_j2, p1, p2, p3, q_12, q_23, q_13)

            dev = 0
            for k1 in range(1, 4):
                for k2 in range(1, 4):
                    dev += partials[k1] * partials[k2] * covariances[(k1, k2)]
            dev = np.sqrt(np.abs(dev))

            p_ki[k] = p1
            dev_ki[k] = dev

            try:
                d_kij[k, worker, w_j1] = partials[1]
                d_kij[k, w_j1, worker] = partials[1]
                d_kij[k, worker, w_j2] = partials[2]
                d_kij[k, w_j2, worker] = partials[2]
                d_kij[k, w_j1, w_j2] = partials[3]
                d_kij[k, w_j2, w_j1] = partials[3]
            except IndexError:
                print(k, worker, w_j1, w_j2)

        # Step 3: Aggregating information from triplets
        # Lemma 4
        if self.debug:
            print('Calculating covariances...')

        cov_kk = np.zeros((l, l))
        for k2 in range(l):
            for k1 in range(k2 + 1):
                w_j1, w_j2 = pairs[k1]
                w_j3, w_j4 = pairs[k2]

                if k1 == k2:
                    cov = dev_ki[k1] * dev_ki[k2]
                else:
                    cov = d_kij[k1, worker, w_j1] * d_kij[k2, worker, w_j3] * self.C(worker, w_j1, w_j3) \
                          + d_kij[k1, worker, w_j1] * d_kij[k2, worker, w_j4] * self.C(worker, w_j1, w_j4) \
                          + d_kij[k1, worker, w_j2] * d_kij[k2, worker, w_j3] * self.C(worker, w_j2, w_j3) \
                          + d_kij[k1, worker, w_j2] * d_kij[k2, worker, w_j4] * self.C(worker, w_j2, w_j4)

                cov_kk[k1, k2] = cov
                cov_kk[k2, k1] = cov
        if self.debug:
            print('Calculating minimal variance...')
        # Calculating a_k for minimal variance of error rate estimates with Lemma 5
        O = np.ones(l)
        try:
            # Least squares or other approximations
            cov_inv = np.linalg.solve(cov_kk, np.identity(cov_kk.shape[0]))
        except LinAlgError:
            cov_inv = np.linalg.pinv(cov_kk)

        B = np.matmul(cov_inv, O)
        A = B / np.linalg.norm(B, ord=1)

        dev = 0
        p = 0
        for i in range(0, l):
            p += A[i] * p_ki[i]
            for j in range(0, l):
                dev += A[i] * A[j] * cov_kk[i, j]
        # TODO: Can we just take the absolute value?
        dev = np.sqrt(np.abs(dev))

        # We want an interval with a given confidence
        t = (1 + confidence) / 2
        # Calculate z_t to give the t'th percentile of the normal distribution
        z_t = stats.norm.ppf(t)
        confidence_interval = z_t * dev

        if self.debug:
            print('Results', p, ',', confidence_interval)
        return np.abs(p), confidence_interval

    def C(self, i: int, j: int, k: int) -> float:
        """
        Method C from Lemma 4 of the paper.
        :param i: worker i.
        :param j: worker j.
        :param k: worker k (j' in the paper).
        :return:
        """
        q_ij = self.dataset.fraction_of_agreement_between(i, j)[0]
        q_ik = self.dataset.fraction_of_agreement_between(i, k)[0]
        q_jk = self.dataset.fraction_of_agreement_between(j, k)[0]
        p_i = self.estimate_error_rate(q_ij, q_ik, q_jk)
        c_ijk = self.dataset.num_shared_samples_for_workers(i, j, k)
        c_ij = self.dataset.num_shared_samples_for_workers(i, j)
        c_jk = self.dataset.num_shared_samples_for_workers(i, k)

        return c_ijk * p_i * (1 - p_i) * (2 * q_jk - 1) / (c_ij * c_jk)

    def generate_peer_pairs_for_worker(self, worker: int, min_shared_samples: int = 1):
        # Step 1: Selecting triples
        # Greedy selection: Sort workers by number of shared tasks with the given worker,
        # then create pairs from the two workers with the most shared tasks with the given
        # worker down to the two workers with the least shared tasks with the given worker.
        sorted_by_shared_samples = self.dataset.get_peers_sorted_by_shared_samples(worker)
        pairs = []
        for i in range(0, len(sorted_by_shared_samples) - 1, 2):
            w1, w2 = sorted_by_shared_samples[i], sorted_by_shared_samples[i + 1]
            if self.dataset.num_shared_samples_for_workers(worker, w1, w2) >= min_shared_samples:
                pairs.append((sorted_by_shared_samples[i], sorted_by_shared_samples[i + 1]))

        return pairs

    def calc_conf_interval(self,
                           worker1: int, worker2: int, worker3: int,
                           p1: float, p2: float, p3: float,
                           q_12: float, q_23: float, q_13: float,
                           confidence=0.9):

        partials = self.calculate_partials_f(q_12, q_13, q_23)
        covariances = self.calc_covariances(worker1, worker2, worker3, p1, p2, p3, q_12, q_23, q_13)

        dev = 0
        for i in range(1, 4):
            for j in range(1, 4):
                dev += partials[i] * partials[j] * covariances[(i, j)]

        # We want an interval with confidence c
        t = (1 + confidence) / 2
        # Calculate z_t to give the t'th percentile of the normal distribution
        z_t = stats.norm.ppf(t)
        confidence_interval = z_t * np.sqrt(dev)

        return confidence_interval

    def calc_covariances(self, worker1: int, worker2: int, worker3: int,
                         p1: float, p2: float, p3: float,
                         q_12: float, q_23: float, q_13: float):
        """Calculates the covariances.

        Returns:
            A dictionary with indices as keys and covariances as values.
        """

        # X_1 := Q_12
        # X_2 := Q_13
        # X_3 := Q_23

        c_12 = self.dataset.num_shared_samples_for_workers(worker1, worker2)
        c_13 = self.dataset.num_shared_samples_for_workers(worker1, worker3)
        c_23 = self.dataset.num_shared_samples_for_workers(worker2, worker3)
        c_123 = self.dataset.num_shared_samples_for_workers(worker1, worker2, worker3)

        # cov_22 = Cov(Q_12, Q_12)
        cov_11 = q_12 * (1 - q_12) / c_12
        # cov_22 = Cov(Q_13, Q_13)
        cov_22 = q_13 * (1 - q_13) / c_13
        # cov_33 = Cov(Q_13, Q_13)
        cov_33 = q_23 * (1 - q_23) / c_23

        # cov_12 = Cov(Q_12, Q_13)
        cov_12 = c_123 * (p1 * (1 - p1) * (2 * q_23 - 1)) / (c_12 * c_13)
        # cov_13 = Cov(Q_12, Q_23)
        cov_13 = c_123 * (p2 * (1 - p2) * (2 * q_13 - 1)) / (c_12 * c_23)
        # cov_23 = Cov(Q_13, Q_23)
        cov_23 = c_123 * (p3 * (1 - p3) * (2 * q_12 - 1)) / (c_13 * c_23)

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
