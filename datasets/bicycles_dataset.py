from . import GroundTruthDataset
import pandas as pd


class BicyclesDataset(GroundTruthDataset):
    def __init__(self, filepath: str):
        dataframe = pd.read_csv(filepath, index_col=0)
        dataframe.rename(columns={
            'image': 'sample',
            'user': 'worker',
        }, inplace=True)

        self._cant_solve = dataframe[dataframe['cant_solve'] == 1]
        self._corrupt_data = dataframe[dataframe['corrupt_data'] == 1]

        # Clean raw dataset from unsolvable and corrupt data and drop unnecessary columns.
        dataframe = dataframe.drop(dataframe[(dataframe['cant_solve'] == 1) | (dataframe['corrupt_data'] == 1)].index)
        dataframe = dataframe.drop(['cant_solve', 'corrupt_data'], axis=1)

        # Drop multiple answers of one worker for one image
        # TODO: Replace with majority vote
        dataframe = dataframe.value_counts().reset_index()\
            .drop_duplicates(subset=['sample', 'worker']).drop(columns=0)

        super().__init__(dataframe)

    @property
    def no_cant_solve(self):
        """Returns the number of samples marked as unsolvable by the workers.

        Returns:
            int: The number of samples marked as unsolvable by the workers.
        """
        return len(self._cant_solve)

    @property
    def no_corrupt_data(self):
        """Returns the number of samples marked as corrupt by the workers.

        Returns:
            int: Number of samples marked as corrupt by the workers.
        """
        return len(self._corrupt_data)

    # def find_peers_for_worker(self, worker: int, peer_count:int=None, min_samples:int=None) -> Tuple[List[int], List[str]]:
    #     """Finds peers for the given worker that worked on the same tasks.

    #     Args:
    #         worker (int): The worker we want to find peers for.
    #         peer_count (int): The maximum number of peers we want to find. None means that min_samples is the
    #         only factor influencing the number of peers. Defaults to None.
    #         min_samples (int): The minimum number of samples that the workers have to share.
    #         None means that peer_count is the only factor influencing the number of peers. Defaults to None.

    #     Returns:
    #         Tuple(List[int], List): A list of the peers and the shared sample names.
    #     """

    #     peers = []
    #     number_shared_images = []
    #     shared_images_names = []
    #     shared_image_names_set = set(
    #         self._grouped_by_worker.get_group(worker)['image'])

    #     for i, frequency in enumerate(self.joint_task_frequencies[worker]):
    #         if peer_count is not None and i > peer_count - 1:
    #             break
    #         else:
    #             images_for_worker = set(
    #                 self._grouped_by_worker.get_group(i)['image'])
    #             shared_image_names_set = shared_image_names_set & images_for_worker

    #             if min_samples is not None and len(shared_image_names_set) < min_samples and len(peers) > 1:
    #                 break

    #             peers.append(i)
    #             number_shared_images.append(len(shared_image_names_set))
    #             shared_images_names.append(list(shared_image_names_set))

    #     if len(peers) < 2:
    #         raise ValueError('Not enough peers found for worker ' + str(worker) + ' that share at least ' + str(min_samples) + ' samples.')

    #     return (peers, shared_images_names[-1])

    # def get_workers_for_sample(self, sample: str):
    #     """Returns an array of the workers that labeled the image with the given image name.

    #     Args:
    #         sample (str): The name of the image.

    #     Returns:
    #         np.ndarray: An array of the workers that labeled the given image.
    #     """
    #     return self._grouped_by_image.get_group(sample)['user'].to_numpy()

    # def get_samples_for_worker(self, worker: int):
    #     """Returns the image names of the images labeled by the given worker.

    #     Args:
    #         worker (int): The worker.

    #     Returns:
    #         np.ndarray: The image names of the images labeled by the given worker.
    #     """
    #     return self._grouped_by_worker.get_group(worker)['image'].to_numpy()

    # def get_votes_for_samples(self, worker: int, samples: np.ndarray):
    #     """Returns the votes of the given worker for the given image/sample names.

    #     Args:
    #         worker (int): The worker the votes of are requested.
    #         samples (np.ndarray): The image names of the images the votes are requested for.

    #     Returns:
    #         np.ndarray: The votes of the given worker for the given image names.
    #     """
    #     return self.df[self.df['image'].isin(samples) & (self.df['user'] == worker)].sort_values(by='image')['answer'].to_numpy()

    # def get_ground_truth_for_samples(self, samples: np.ndarray):
    #     """Returns the ground truth for the given image names.

    #     Args:
    #         samples (np.ndarray): The images names of the images we want the ground truth for.

    #     Returns:
    #         np.ndarray: The ground truth for the given images ordered by image name.
    #     """
    #     return self.df[self.df['image'].isin(samples)].sort_values(by='image').drop_duplicates(subset='image')['is_bicycle']

    # def get_measured_error_rate_for_worker(self, worker: int, samples = None):
    #     if samples is None:
    #         samples = self.get_samples_for_worker(worker)

    #     truth = self.get_ground_truth_for_samples(samples)
    #     votes = self.get_votes_for_samples(worker, samples)
    #     error = np.count_nonzero(truth != votes) / len(truth)
    #     return error

    # def get_measured_error_rate_for_worker(self, worker: int):
    #     samples_for_worker = self.get_samples_for_worker(worker)
    #     votes = self.get_votes_for_samples(worker, samples_for_worker)
    #     ground_truth = self.get_ground_truth_for_samples(samples_for_worker)

    #     diff = np.where(votes == ground_truth, 0, 1)
    #     return np.count_nonzero(diff) / len(diff)

    # def get_measured_error_rate_for_workers(self, workers: Union[List[int], np.ndarray]) -> np.ndarray:
    #     error_rates = []
    #     for worker in workers:
    #         error_rates.append(self.get_measured_error_rate_for_worker(worker))

    #     return np.array(error_rates)

    # def get_measured_error_rate_for_superworker(self, superworker: Union[List[int], np.ndarray], samples: Union[List[str], np.ndarray]):
    #     truth = self.get_ground_truth_for_samples(samples)
    #     votes = []
    #     for worker in superworker:
    #         votes.append(self.get_votes_for_samples(worker, samples))
    #     votes = np.array(votes)
    #     vote = majority_vote(votes)
    #     error = np.count_nonzero(np.where(truth == vote, 0, 1)) / len(truth)
    #     return error

    # def get_ground_truth_for_samples(self, samples: np.ndarray):
    #     d = self.df.drop_duplicates(subset='image', inplace=False)
    #     return d[d['image'].isin(samples)].sort_values('image')['is_bicycle']


