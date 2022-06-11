import numpy as np
from typing import List, Tuple

from . import GroundTruthDataset
import pandas as pd


class CIFAR10HDataset(GroundTruthDataset):
    def __init__(self,
                 filepath: str,
                 categories: Tuple[List[str], List[str]] = (['cat'], ['dog']),
                 prefilter_mode: str = None,
                 prefilter_threshold: float = 0.4,
                 debug: bool = False):
        df = pd.read_csv(filepath)
        df.rename(columns={
            'image_filename': 'sample',
            'annotator_id': 'worker',
            'true_category': 'truth',
            'chosen_category': 'answer'
        }, inplace=True)

        # Calculate error rates from attention checks
        self.attn_check_error_rates = {}
        worker_df = df[df['is_attn_check'] == True].groupby('worker')

        for worker in df['worker'].unique():
            attention_check_error_rate = \
                1.0 - np.count_nonzero(worker_df.get_group(worker)['correct_guess']) / len(worker_df.get_group(worker))
            self.attn_check_error_rates[worker] = attention_check_error_rate

            if prefilter_mode == "attention_checks" and attention_check_error_rate > prefilter_threshold:
                df = df[df['worker'] != worker]

        df = df[
            (df['truth'].isin(categories[0])) |
            (df['truth'].isin(categories[1]))]

        df['truth'] = (df['truth'].isin(categories[0])).astype(int)
        df['answer'] = (df['answer'].isin(categories[0])).astype(int)
        print(df.head())

        # df.drop(df[df['is_attn_check'] == 1].index, inplace=True)
        df.drop(columns=['reaction_time', 'time_elapsed', 'cifar10_test_test_idx'],
                inplace=True)

        super().__init__(df, prefilter_mode=prefilter_mode, prefilter_threshold=prefilter_threshold, debug=debug)

    def get_attention_check_error_rate(self, worker: int) -> float:
        """
        Returns the error rate the given worker achieved with attention check samples.
        :param worker: The worker the error rate should be calculated for.
        :return: The error rate the given worker achieved with attention check samples.
        """
        return self.attn_check_error_rates[worker]
