from itertools import combinations
import math
from typing import List, Tuple
import numpy as np
import cv2
from scipy import stats

def wilson_score(confidence: float, p_est: float, n: int):
    """
    Calculates the confidence half size using the Wilson Score.
    :param confidence: The required confidence level.
    :param p_est: The estimated error rate.
    :param n: The number of samples.
    :return: The confidence interval half size.
    """
    t = (1 + confidence) / 2
    z_t = stats.norm.ppf(t)

    numerator = z_t * np.sqrt(p_est * (1 - p_est) / n + z_t ** 2 / (4 * n ** 2))
    denominator = 1 + 1 / n * z_t ** 2
    conf = numerator / denominator
    return conf


def std_score(confidence: float, p_est: float, n: int):
    """
    Calculates the confidence half size using the classic confidence interval estimation
    for the standard normal distribution
    :param confidence: The required confidence level.
    :param p_est: The estimated error rate.
    :param n: The number of samples.
    :return: The confidence interval half size.
    """
    # Confidence calculation
    t = (1 + confidence) / 2
    z_t = stats.norm.ppf(t)

    conf = z_t * np.sqrt((p_est * (1 - p_est)) / n)
    return conf


def majority_vote(votes):
    """Calculates the majority vote from the votes of m given workers.

    Args:
        votes (np.ndarray): A mxn matrix where m is the number of workers
        and n is the number of votes each worker makes.

    Returns:
        np.ndarray: A vector of length n containing the majority vote.
        :param votes:
        :return:
    """
    return np.sum(votes, axis=0) / len(votes) > 0.5


def weighted_majority_vote(votes, weights):
    """Calculates the majority vote from the votes of m given workers.

    :param votes:  A mxn matrix where m is the number of workers and n is the number of votes each worker makes.
    :param weights: The weights for each of the workers.
    :return: A vector of length n containing the majority vote.
    """
    weighted_votes = (weights * votes.T).T
    return np.sum(weighted_votes, axis=0) / np.sum(weights, axis=0) > 0.5


def calculate_brightness_level_for_image(image: np.ndarray):
    """Calculates the brightness levels for the given image.

    Args:
        image (np.ndarray): The image.

    Returns:
        Tuple[float, float, float]: mean, min, max brightness values for the given image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.mean(), gray.std(), gray.min(), gray.max()


def generate_superworkers_exhaustively(worker_indices: List[int]) -> List[Tuple[List[int], List[int]]]:
    """Generates all possible superworker combinations for the given workers.

        Args:
            worker_indices (List[int]): The workers out of which we want to construct superworkers.

        Returns:
            List[Tuple[List[int], List[int]]]: A list of superworker tuples, each superworker being a list
            of the workers that construct the superworker.
        """
    size = len(worker_indices)
    half_size = math.ceil(len(worker_indices) * 0.5)
    superworker_groups = []
    for i in range(1, half_size + 1):
        lefts = list(combinations(worker_indices, i))
        rights = list(combinations(worker_indices, size - i))
        superworker_groups.extend(
            [(left, right) for left in lefts for right in rights if set(left).isdisjoint(set(right))])
    return superworker_groups
