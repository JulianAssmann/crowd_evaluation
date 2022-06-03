import numpy as np
from typing import List

from . import GroundTruthDataset
import pandas as pd


class CIFAR10HDataset(GroundTruthDataset):
    def __init__(self, filepath: str, binary_categories: List = None):
        df = pd.read_csv(filepath)
        df.rename(columns={
            'image_filename': 'sample',
            'annotator_id': 'worker',
            'true_category': 'truth',
            'chosen_category': 'answer'
        }, inplace=True)

        if binary_categories is not None:
            df = df[(df['truth'] == binary_categories[0]) | (df['truth'] == binary_categories[1])]

        # Calculate error rates from attention checks
        self.attn_check_error_rates = {}
        worker_df = df[df['is_attn_check'] == True].groupby('worker')
        for worker in df['worker'].unique():
            self.attn_check_error_rates[worker] = \
                1.0 - np.count_nonzero(worker_df.get_group(worker)['correct_guess']) / len(worker_df.get_group(worker))

        # df.drop(df[df['is_attn_check'] == 1].index, inplace=True)
        df.drop(columns=['reaction_time', 'time_elapsed', 'cifar10_test_test_idx'],
                inplace=True)

        super().__init__(df)

    def get_attention_check_error_rate(self, worker: int) -> float:
        """
        Returns the error rate the given worker achieved with attention check samples.
        :param worker: The worker the error rate should be calculated for.
        :return: The error rate the given worker achieved with attention check samples.
        """
        return self.attn_check_error_rates[worker]
