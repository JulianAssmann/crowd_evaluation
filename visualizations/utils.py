import numpy as np
from typing import List, Union

from crowd_evaluation import ConfidenceEvaluatorNew, ConfidenceEvaluatorOld
from datasets import Dataset, GroundTruthDataset, SyntheticDataset
import plotly.graph_objects as go
import os


def visualize_error_rates(dataset: Dataset,
                          workers: Union[List[int], np.ndarray],
                          p_ests: List[np.ndarray],
                          confs: List[np.ndarray],
                          labels: List[str],
                          show_true: bool = True,
                          show_measured: bool = True,
                          title: str = None,
                          filename: str = None):
    assert (len(p_ests) == len(confs))
    assert (len(confs) == len(labels))

    x_labels = np.char.add(np.full(len(workers), "W"), np.char.mod('%d', workers + 1))

    p_true = None
    p_measured = None
    if issubclass(type(dataset), GroundTruthDataset):
        if show_true:
            p_true = dataset.get_true_error_rates_for_workers(dataset.workers) * 100
            # p_true_text = []
            # for p_tt in p_true:
            #     p_true_text.append("{:.2f}".format(p_tt))
        if show_measured:
            p_measured = dataset.get_measured_error_rates_for_workers(dataset.workers) * 100

    # p_ests_text = []
    # for i, p_tt in enumerate(p_ests * 100):
    #     p_ests_text.append("{:.2f}".format(p_tt) + '<br>±' + "{:.2f}".format(confs[i]*100))
    #
    # p_ests_wilson_text = []
    # for i, p_tt in enumerate(p_ests_wilson * 100):
    #     p_ests_wilson_text.append("{:.2f}".format(p_tt) + '<br>±' + "{:.2f}".format(confs_wilson[i]*100))

    fig = go.Figure()
    if p_true is not None:
        fig.add_trace(go.Bar(y=p_true,
                             name=r'$p_{true}$',
                             x=x_labels))

    if p_measured is not None:
        fig.add_trace(go.Bar(y=p_measured,
                             name=r'$p_{measured}$',
                             x=x_labels))

    for i in range(len(p_ests)):
        p_est = p_ests[i]
        conf = confs[i]
        label = labels[i]

        fig.add_trace(go.Bar(y=p_est * 100,
                             name=label,
                             x=x_labels,
                             error_y=dict(type='data', array=conf * 100) if conf is not None else None))

    margin_top = 5 if title is None else 40
    fig.update_layout(
        barmode='group',
        title=title,
        xaxis_title="Worker",
        yaxis_title="Error rate in %",
        margin=dict(r=5, l=5, t=margin_top, b=5))
    fig.show()
    if filename is not None:
        fig.write_image(filename, scale=5)


def say(msg="Finish", voice="Victoria"):
    os.system(f'say -v {voice} {msg}')
