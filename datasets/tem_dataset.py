from datasets import GroundTruthDataset
import pandas as pd

class TemDataset(GroundTruthDataset):
    def __init__(self, path: str, prefilter_mode: str = None, prefilter_threshold: float = 0.4, debug: bool = False):
        df = pd.read_csv(path, delimiter='\t')
        df.rename(columns={'!amt_annotation_ids': 'worker', 'orig_id': 'sample', 'gold': 'truth', 'response': 'answer'},
                  inplace=True)
        df.drop(columns=['!amt_worker_ids'], inplace=True)
        worker_names_to_int_map = {w: i for (i, w) in enumerate(df['worker'].unique())}
        df = df.replace({'worker': worker_names_to_int_map})

        super().__init__(df, prefilter_mode, prefilter_threshold, debug)
