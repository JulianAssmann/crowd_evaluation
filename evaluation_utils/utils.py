import numpy as np
from typing import List, Union


def calculate_estimation_accuracy(p_true: Union[np.ndarray, List[float]],
                                  p_est: Union[np.ndarray, List[float]],
                                  confs: Union[np.ndarray, List[float]]) -> float:
    """Calculates the accuracy of the estimation given the true error rate and the results from an evaluator.

    :param p_true: The error rates that serve as the true error rates
    :param p_est: The error rates estimated by the evaluator
    :param confs: The confidence interval half sizes estimated by the evaluator
    :return: The accuracy of the estimation, i.e. whether the true error rate is inside the confidence interval
    """
    assert (p_true.shape == p_est.shape)
    assert (p_est.shape == confs.shape)

    min_limit, max_limit = p_est-confs, p_est+confs
    inside_interval = np.where((min_limit <= p_true) & (p_true <= max_limit), 1, 0)
    return np.count_nonzero(inside_interval) / len(p_true)
