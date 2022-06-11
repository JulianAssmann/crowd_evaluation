from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from itertools import combinations
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class Dataset(ABC):
    def __init__(self, dataframe: pd.DataFrame,
                 prefilter_mode: str = None,
                 prefilter_threshold: float = 0.4,
                 debug: bool = False):
        """

        :param dataframe: The data of the dataset.
        dataframe should include the following columns:
        - worker
        - sample
        - answer
        - truth (optional)

        :param prefilter_mode: The prefilter mode to filter out obvious spammers.
        Possible values are None, majority_vote and truth (if a ground truth is available)

        :param prefilter_threshold: The threshold error rate above which a worker is regarded as a spammer and excluded.
        """
        self._df = dataframe
        self.debug = debug

        if prefilter_mode == 'majority_vote' or prefilter_mode == 'truth':
            self._df = self._filter(self._df, prefilter_mode, prefilter_threshold)

        self._workers = self._df.sort_values(by='worker')['worker'].unique()
        self._samples = self._df.sort_values(by='sample')['sample'].unique()

        # Create dataframe that stores shared samples for each 2-worker combination
        workers_to_samples = self._df.groupby('worker')['sample'].agg(set)
        self._workers_shared_samples = {}
        self._num_workers_shared_samples = {}
        for (i, j) in combinations(workers_to_samples.index, 2):
            key = frozenset([i, j])
            shared_samples = workers_to_samples[i].intersection(workers_to_samples[j])
            self._workers_shared_samples[key] = shared_samples
            self._num_workers_shared_samples[key] = len(shared_samples)

        self._calculate_joint_task_frequencies()

        # Stores fractions of agreement between different worker pairs for caching purposes.
        self._agreement_fractions = pd.DataFrame(columns=['workers', 'agreement_fraction', 'num_samples'])

        # Calculates the majority votes for each sample.
        self._calculate_majority_votes()

        # Cache dataset grouped by worker_sample for later use
        self._grouped_by_sample = self._df.groupby('sample')
        self._grouped_by_worker = self._df.groupby('worker')

    def _filter(self, df: pd.DataFrame, prefilter_mode='majority_vote', prefilter_threshold=0.4) -> pd.DataFrame:
        bad_workers = []

        if prefilter_mode == 'majority_vote':
            f = lambda x: x.mode().iat[0]
            majority_vote_df = df.groupby('sample')['answer'].apply(f).reset_index(name=prefilter_mode)
            base_df = pd.merge(df, majority_vote_df, on='sample')
        else:
            base_df = df

        for worker in base_df['worker'].unique():
            worker_df = base_df[base_df['worker'] == worker]
            fraction_of_disagreement = np.count_nonzero(worker_df['answer'] - worker_df[prefilter_mode]) / len(
                worker_df)
            if fraction_of_disagreement > prefilter_threshold:
                bad_workers.append(worker)

        if self.debug:
            print('Bad workers excluded:', bad_workers)

        df = df[~df['worker'].isin(bad_workers)]
        return df

    def _calculate_majority_votes(self) -> None:
        self._majority_votes = {}
        for sample in self.samples:
            sample_df = self.df[self.df['sample'] == sample]
            majority_vote = sample_df['answer'].value_counts().index[0]
            self._majority_votes[sample] = majority_vote

    def _calculate_joint_task_frequencies(self) -> None:
        # df_selected_worker serves as reference, i.e. the index
        df_relevant_columns = self.df[['sample', 'worker']]

        df_selected_worker = df_relevant_columns.rename(
            columns={"worker": "other"})

        # Join reference with all the other users
        df_joint = pd.merge(df_selected_worker,
                            df_relevant_columns, on="sample")

        # Make a pivot table that contains the joint answer counts
        df_joint = df_joint.pivot_table(
            index="other", columns="worker", values="sample", aggfunc="count").fillna(0)

        self._joint_task_frequencies = df_joint.astype('int32')

    @property
    def categories(self) -> np.ndarray:
        return self._df['answer'].unique().to_numpy()

    @property
    def df(self) -> pd.DataFrame:
        """The underlying pandas dataframe."""
        return self._df

    @property
    def num_workers(self) -> int:
        """The number of workers in this dataset.

        :return: The number of workers in this dataset.
        """
        return len(self._workers)

    @property
    def workers(self) -> np.ndarray:
        """All workers in this dataset.

        :return: All workers in this dataset.
        """
        return np.sort(self._workers)

    @property
    def num_samples(self) -> int:
        """The number of samples in this dataset.

        :return: The number of samples in this dataset.
        """
        return len(self._samples)

    @property
    def samples(self):
        """All samples in this dataset.

        :return: All the samples in this dataset.
        """
        return self._samples

    def fraction_of_agreement_between(self, worker1: int, worker2: int) -> Tuple[float, int, List]:
        """The fraction of agreement between worker1 and worker2 for the samples both annotated.

        :return: A tuple of form
        (fraction of agreement between worker1 and worker2,
        number of shared samples between worker1 and worker2,
        shared samples between worker1 and worker2)
        """
        s = frozenset([worker1, worker2])
        if s in self._agreement_fractions['workers']:
            return self._agreement_fractions[self._agreement_fractions['workers'] == s]['agreement_fraction']

        shared_samples = self.shared_samples_for_workers(worker1, worker2)

        fraction_of_agreement = 0.0
        if len(shared_samples) == 0:
            fraction_of_agreement = 1.0
        else:
            worker1_answers = self.df[
                (self.df['worker'] == worker1) & (self.df['sample'].isin(shared_samples))] \
                .sort_values('sample')['answer'].to_numpy()
            worker2_answers = self.df[
                (self.df['worker'] == worker2) & (self.df['sample'].isin(shared_samples))] \
                .sort_values('sample')['answer'].to_numpy()

            nonzero = np.count_nonzero(worker1_answers == worker2_answers)
            fraction_of_agreement = nonzero / float(len(shared_samples))

        self._agreement_fractions = self._agreement_fractions.append({
            'workers': s,
            'agreement_fraction': fraction_of_agreement,
            'num_samples': len(shared_samples),
        },
            ignore_index=True)

        return fraction_of_agreement, len(shared_samples), shared_samples

    def shared_samples_for_workers(self, worker1: int, worker2: int, worker3: int = None) -> List:
        """The samples worker1, worker2 and potentially worker3 (if not None) all annotated (shared samples).

        :return: The shared samples between worker1, worker2 and potentially worker 3 (if not None).
        """

        w1w2_shared = self._workers_shared_samples[frozenset([worker1, worker2])]

        if worker3 is None:
            return w1w2_shared
        else:
            w2w3_shared = self._workers_shared_samples[frozenset([worker2, worker3])]
            return w1w2_shared.intersection(w2w3_shared)

    def num_shared_samples_for_workers(self, worker1: int, worker2: int, worker3: int = None) -> int:
        """The number of samples worker1, worker2 and potentially worker3 (if not None) all annotated (shared samples).

        :return: The number of shared samples between worker1, worker2 and potentially worker 3 (if not None).
        """
        return len(self.shared_samples_for_workers(worker1, worker2, worker3))

    def get_peers_sorted_by_shared_samples(self, worker: int):
        """Sorts the workers by the number of shared samples with the given worker (descending).

        :param worker: The worker we want peers for.
        :return: A  list of all workers sorted by the number of shared samples with the given worker (descending).
        """

        sorted_by_task_frequency = self._joint_task_frequencies[worker].sort_values(ascending=False)
        # Only return workers with at least one shared sample
        sorted_by_task_frequency = sorted_by_task_frequency[sorted_by_task_frequency > 0]
        sorted_by_task_frequency = np.array(sorted_by_task_frequency.index)

        # Delete worker from the sorted list as it cannot be its own peer
        workers_list = np.delete(sorted_by_task_frequency, np.where(sorted_by_task_frequency == worker), axis=0)

        return workers_list

    def find_peers_for_worker(self, worker: int,
                              peer_count: Optional[int] = None,
                              min_samples: Optional[int] = 0) -> Tuple[List[int], List[str]]:
        """
        Finds peers for the given worker that worked on the same tasks.

        :param worker: The worker we want to find peers for.
        :param peer_count: The maximum number of peers we want to find. None means that min_samples is the
        only factor influencing the number of peers. Defaults to None.
        :param min_samples: The minimum number of samples that the workers have to share.
        None means that peer_count is the only factor influencing the number of peers. Defaults to None.

        :return: A tuple containing the list of the peers and the shared samples,
         or empty lists when not enough (<2) peers were found that shared at least min_samples samples.
        """

        peers = []
        number_shared_samples = []
        shared_samples = []
        shared_samples_set = set(
            self._grouped_by_worker.get_group(worker)['sample'])

        for i, frequency in enumerate(self._joint_task_frequencies[worker].sort_values(ascending=False)):
            if i == worker:
                continue
            if peer_count is not None and i > peer_count - 1:
                break
            else:
                samples_for_worker = set(
                    self._grouped_by_worker.get_group(i)['sample'])
                new_shared_samples_set = shared_samples_set & samples_for_worker

                if len(new_shared_samples_set) <= 0:
                    break

                shared_samples_set = new_shared_samples_set
                if min_samples is not None and len(shared_samples_set) < min_samples and len(peers) > 1:
                    break

                peers.append(i)
                # number_shared_samples.append(len(shared_samples_set))
                # shared_samples.append(list(shared_samples_set))

        if len(peers) < 2:
            return [], []
            # raise ValueError('Not enough peers found for worker ' + str(worker) + ' that share at least ' + str(
            #     min_samples) + ' samples.')

        return peers, list(shared_samples_set)

    def get_samples_for_worker(self, worker: int):
        """Returns the samples labeled by the given worker.

        :param worker: The worker.

        :return: The samples labeled by the given worker.
        """
        return self.df[self.df['worker'] == worker]['sample']

    def get_answers(self, worker: int, samples: Union[List, np.ndarray]):
        """
        Returns the votes of the given worker for the given samples ordered by the samples.

        :param worker: The worker the votes of are requested.
        :param samples: The sample names the votes of are requested.

        :return: The votes of the given worker for the given samples ordered by the samples.
        """
        return self.df[(self.df['worker'] == worker) & (self.df['sample'].isin(samples))]['answer']

    def get_answers_for_sample(self, workers: Union[List, np.ndarray], sample) -> dict:
        """
        Returns the answers of the given workers for the given sample.

        :param workers: The worker the vote of is requested.
        :param sample: The sample the vote of is requested.

        :return: The votes of the workers for the given sample as a dictionary, where the workers are the keys
        and the values are the answers of the corresponding workers.
        """

        sample_answers = self.df[(self.df['worker'].isin(workers)) & (self.df['sample'] == sample)].set_index('worker')
        return sample_answers['answer'].to_dict()

    def get_majority_vote_for_samples(self, samples: Union[List, np.ndarray]) -> np.ndarray:
        """
        Returns the majority votes for the given samples.
        :param samples: The samples the majority votes are requested for.
        :return: The majority votes for the given samples.
        """
        return np.array([self._majority_votes[sample] for sample in samples])

    def get_workers_for_sample(self, sample) -> np.ndarray:
        """
        Returns the workers that worked on the given sample.
        :param sample: The sample.
        :return: The workers that worked on the given sample.
        """
        return self.df[self.df['sample'] == sample]['worker']
