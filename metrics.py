# import
import numpy as np
# confusion matrix
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# General
from sklearn.metrics import classification_report
# ROC curve and metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
# PRC curve and metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay

# paint
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib import rcParams
# plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import plot, iplot, init_notebook_mode

rcParams["font.size"] = 15  # 设置字体大小
palette = pyplot.get_cmap("tab10")

# our method
from .simpletools import ListArray_2_FlattenArray

config_plotly = {
    "toImageButtonOptions": {
        "format": "svg",  # one of png, svg, jpeg, webp
        "filename": "custom_image",
        #     'height': 500,
        #     'width': 700,
        "scale": 1,  # Multiply title/legend/axis/canvas sizes by this factor
    },
    "scrollZoom": True,
    "modeBarButtonsToAdd": [
        "drawline",
        "drawopenpath",
        "drawclosedpath",
        "drawcircle",
        "drawrect",
        "eraseshape",
    ],
    # 'dragmode':'drawopenpath',
    # 'newshape_line_color':'cyan',
}


def paint_general_cm(
    list_targets_clfs, list_prediction_clfs, list_display_labels=["Not", "Yes"], titles_subplot=None, list_camps=None, path_save=None
):
    """
    Input: list_targets_clfs, list_prediction_clfs, list_display_labels=["Not", "Yes"], titles_subplot=None, list_camps=None, path_save=None
    Paint: 1*len(list_targets_clfs) matrix and 1 general matrix
    """

    num_cms = len(list_targets_clfs) + 1
    fig, axes = plt.subplots(1, num_cms, figsize=(
        6 * num_cms + (num_cms - 1), 5))
    for idx, (target, prediction) in enumerate(
        zip(list_targets_clfs, list_prediction_clfs)
    ):
        if list_camps:
            ConfusionMatrixDisplay.from_predictions(
                target,
                prediction,
                ax=axes[idx],
                display_labels=list_display_labels,
                cmap=list_camps[idx],
            )
        else:
            ConfusionMatrixDisplay.from_predictions(
                target,
                prediction,
                ax=axes[idx],
                display_labels=list_display_labels,
                cmap=plt.cm.Blues,
            )
    all_targets = ListArray_2_FlattenArray(list_targets_clfs)
    all_predictions = ListArray_2_FlattenArray(list_prediction_clfs)
    ConfusionMatrixDisplay.from_predictions(
        all_targets,
        all_predictions,
        ax=axes[-1],
        display_labels=list_display_labels,
        cmap=plt.cm.Reds,
    )
    if titles_subplot:
        for idx, title in enumerate(titles_subplot):
            axes[idx].set_title(title)
    if path_save:
        pyplot.savefig(path_save, format='svg',
                       dpi=500, bbox_inches='tight', pad_inches=0.1)


def paint_general_roc_pr(list_targets_clfs, list_pros_clfs, list_name_clfs=None, path_save=None):
    """
    Input: list_targets_clfs, list_pros_clfs, list_name_clfs=None, path_save=None
    Paint: draw roc and prc, each curve have multiple curve with different fold
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    if not list_name_clfs:
        list_name_clfs = [
            f"Fold-{idx}" for idx in range(len(list_targets_clfs))]
    for idx, (target, pro) in enumerate(zip(list_targets_clfs, list_pros_clfs)):
        RocCurveDisplay.from_predictions(
            target, pro, ax=axes[0], name=list_name_clfs[idx]
        )
        PrecisionRecallDisplay.from_predictions(
            target, pro, ax=axes[1], name=list_name_clfs[idx]
        )
    all_targets = ListArray_2_FlattenArray(list_targets_clfs)
    all_pros = ListArray_2_FlattenArray(list_pros_clfs)
    RocCurveDisplay.from_predictions(
        all_targets, all_pros, ax=axes[0], name="General")
    PrecisionRecallDisplay.from_predictions(
        all_targets, all_pros, ax=axes[1], name="General"
    )
    if path_save:
        pyplot.savefig(path_save, format='svg',
                       dpi=500, bbox_inches='tight', pad_inches=0.1)


def get_general_auc(list_targets_clfs, list_pros_clfs):
    """
    Input: list_targets_clfs, list_pros_clfs
    return: [mean_roc, std_roc, mean_prc, std_prc]
    """
    auc_roc, ap_pr = [], []
    for target, pro in zip(list_targets_clfs, list_pros_clfs):
        auc_roc.append(roc_auc_score(target, pro))
        ap_pr.append(average_precision_score(target, pro))
    mean_roc, std_roc = np.mean(auc_roc), np.std(auc_roc)
    mean_prc, std_prc = np.mean(ap_pr), np.std(ap_pr)
    print(
        "The CV Auc-ROC: {:.4f} +/- {:.4f}, the CV AP-PR: {:.4f} +/- {:.4f}".format(
            mean_roc, std_roc, mean_prc, std_prc
        )
    )
    return [mean_roc, std_roc, mean_prc, std_prc]


def paint_prob_distribution(
    list_targets_clfs, list_pros_clfs, group_labels=["Not", "Yes"]
):
    """
    Input: list_targets_clfs, list_pros_clfs, group_labels=['Not','Yes']
    Paint: probability distribution of different labels
    """
    hist_data = [
        list_pros_clfs[list_targets_clfs == 0],
        list_pros_clfs[list_targets_clfs == 1],
    ]
    fig = ff.create_distplot(hist_data, group_labels,
                             bin_size=0.01, show_curve=True)
    fig.update_layout(title_text="Score Distribution of different group")
    fig.show(config=config_plotly)
