# basic
import numpy as np
import pandas as pd
from tqdm import tqdm
# preprocess
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
# model
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
# ROC curve and metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
# PRC curve and metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score, make_scorer, f1_score
# paint
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib import rcParams
rcParams['font.size'] = 15  # 设置字体大小
palette = pyplot.get_cmap('tab10')
# pyplot.style.use('seaborn-white')

# our method
from .metrics import (
    get_general_auc,
    get_metric_mul,
    paint_general_cm,
    paint_cm_mul,
    paint_general_roc_pr,
    paint_prob_distribution,
)
from .simpletools import cal_CI
from .simpletools import ListArray_2_FlattenArray


def clf_train_test_cv(
    data,
    target,
    cv,
    clf,
    target_names=["Neg", "Pos"],
    list_features_cat=None,
    show_cm=1,
    show_roc_pr=1,
    show_score_dis=1,
    show_not=0,
):
    """
    Input: 
        data, target, cv, clf, preprocessor=None, target_names=['Tra','Not'], list_features_cat=None, show_cm=1, show_roc_pr=1, show_score_dis=1, show_not=0
    Conduct: 
        model development and performance
    Paint: 
        the required curve as the input
    Return: 
        performance, clf, list_data_clfs, list_target_clfs, list_prob_clfs, list_pred_clfs
    """
    print(f"Data    count: {data.shape[0]}")
    print(f"Feature count: {data.shape[1]}")
    print(f"CV:  {cv}")
    if show_not == 1:
        show_each, show_cm, show_roc_pr, show_score_dis = 0, 0, 0, 0
    for idx, (idx_train, idx_test) in enumerate(tqdm(cv.split(data, target))):
        if isinstance(data, pd.DataFrame):
            data_train, data_test = data.iloc[idx_train], data.iloc[idx_test]
        else:
            data_train, data_test = data[idx_train], data[idx_test]
        target_train, target_test = target[idx_train], target[idx_test]
        # model fit and predict
        if isinstance(clf, CatBoostClassifier):
            clf.fit(
                data_train,
                target_train,
                cat_features=list_features_cat,
                verbose=False,
                plot=False,
            )
        elif isinstance(clf, LGBMClassifier):
            clf.fit(data_train, target_train,
                    categorical_feature=list_features_cat)
        else:
            clf.fit(data_train, target_train)
        clf_prediction = clf.predict(data_test)
        clf_pro = clf.predict_proba(data_test)[:, 1]
        # if show_each:
        #     print(classification_report(target_test, clf_prediction, target_names=target_names))
        #     print("-------------------------------------------------------------------------------")
        if idx == 0:
            list_data_clfs, list_target_clfs, list_prob_clfs, list_pred_clfs = (
                [data_test],
                [target_test],
                [clf_pro],
                [clf_prediction],
            )
        else:
            list_data_clfs.append(data_test)
            list_target_clfs.append(target_test)
            list_prob_clfs.append(clf_pro)
            list_pred_clfs.append(clf_prediction)
    # performance = [mean_roc, std_roc, mean_prc, std_prc]
    performance = get_general_auc(list_target_clfs, list_prob_clfs)

    # refit with all data
    if isinstance(clf, CatBoostClassifier):
        clf.fit(data, target, cat_features=list_features_cat,
                verbose=False, plot=False)
    elif isinstance(clf, LGBMClassifier):
        clf.fit(data, target, categorical_feature=list_features_cat)
    else:
        clf.fit(data, target)

    # show different chart
    if show_cm:
        paint_general_cm(list_target_clfs, list_pred_clfs, list_display_labels=target_names)
    if show_roc_pr:
        paint_general_roc_pr(list_target_clfs, list_prob_clfs)

    list_target_clfs, list_prob_clfs, list_pred_clfs = map(
        ListArray_2_FlattenArray, [
            list_target_clfs, list_prob_clfs, list_pred_clfs]
    )
    if show_score_dis:
        paint_prob_distribution(list_target_clfs, list_prob_clfs)

    return (
        performance,
        clf,
        list_data_clfs,
        list_target_clfs,
        list_prob_clfs,
        list_pred_clfs,
    )

def clf_train_test_cv_mul(
    data,
    target,
    cv,
    clf,
    target_names=None,
    list_features_cat=None,
    show_cm=1,
):
    """
    Input: 
        data, target, cv, clf, preprocessor=None, target_names=['Tra','Not'], list_features_cat=None, show_cm=1, show_roc_pr=1, show_score_dis=1, show_not=0
    Conduct: 
        model development and performance
    Paint: 
        the required curve as the input
    Return: 
        performance, clf, list_data_clfs, list_target_clfs, list_prob_clfs, list_pred_clfs
    """
    print(f"Data    count: {data.shape[0]}")
    print(f"Feature count: {data.shape[1]}")
    print(f"CV:  {cv}")
    for idx, (idx_train, idx_test) in enumerate(tqdm(cv.split(data, target))):
        if isinstance(data, pd.DataFrame):
            data_train, data_test = data.iloc[idx_train], data.iloc[idx_test]
        else:
            data_train, data_test = data[idx_train], data[idx_test]
        target_train, target_test = target[idx_train], target[idx_test]
        # model fit and predict
        if isinstance(clf, CatBoostClassifier):
            clf.fit(
                data_train,
                target_train,
                cat_features=list_features_cat,
                verbose=False,
                plot=False,
            )
        elif isinstance(clf, LGBMClassifier):
            clf.fit(data_train, target_train,
                    categorical_feature=list_features_cat)
        else:
            clf.fit(data_train, target_train)
        clf_prediction = clf.predict(data_test)
        clf_pro = clf.predict_proba(data_test)
        if idx == 0:
            list_data_clfs, list_target_clfs, list_prob_clfs, list_pred_clfs = (
                [data_test],
                [target_test],
                [clf_pro],
                [clf_prediction],
            )
        else:
            list_data_clfs.append(data_test)
            list_target_clfs.append(target_test)
            list_prob_clfs.append(clf_pro)
            list_pred_clfs.append(clf_prediction)
    # performance = [mean_roc, std_roc, mean_prc, std_prc]
    performance = get_metric_mul(list_target_clfs, list_prob_clfs)

    # refit with all data
    if isinstance(clf, CatBoostClassifier):
        clf.fit(data, target, cat_features=list_features_cat,
                verbose=False, plot=False)
    elif isinstance(clf, LGBMClassifier):
        clf.fit(data, target, categorical_feature=list_features_cat)
    else:
        clf.fit(data, target)

    # show different chart
    if show_cm:
        paint_cm_mul(list_target_clfs, list_pred_clfs, list_display_labels=target_names)

    list_target_clfs, list_prob_clfs, list_pred_clfs = map(
        ListArray_2_FlattenArray, [
            list_target_clfs, list_prob_clfs, list_pred_clfs]
    )

    return (
        performance,
        clf,
        list_data_clfs,
        list_target_clfs,
        list_prob_clfs,
        list_pred_clfs,
    )

def clf_test_external(df_test, target_test, clf, name_clf='clf', show_score_dis=0, show_not=0,  name_target=['Negative', 'Positive'], preprocessor=None, flag_print=None):
    '''
    test the fitted model in external validation dataset
    Input： df_test, clf, show_score_dis=0, show_not=0, col_target='tracheotomy', target_names=['Not','Tra']
    Output: the dataset shape and the performance (auroc, auprc)
    Paint: if set, the score distribution, confusion matrix, roc, prc
    Return: clf_pro, clf_prediction, auc_roc, auc_pr
    '''
    # train each combination
    data_test = df_test if not isinstance(clf, XGBClassifier) else df_test.values
    if flag_print:
        print(f"The testing dataset:{data_test.shape}")

    if 'SVM' in name_clf and preprocessor:
        print('SVM for preprocessor')
        df_test = preprocessor.transform(df_test)

    # model prediction
    clf_prediction = clf.predict(data_test)
    clf_pro = clf.predict_proba(data_test)[:, 1]

    if show_score_dis:
        paint_prob_distribution(target_test, clf_pro)
    auc_roc = roc_auc_score(target_test, clf_pro)
    auc_pr = average_precision_score(target_test, clf_pro)
    if flag_print:
        print(
            "The General AUC-ROC: {:.4f}, the General AP-PR: {:.4f}".format(auc_roc, auc_pr))

    if show_not == 0:
        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        ConfusionMatrixDisplay.from_predictions(
            target_test, clf_prediction, ax=axes[0], display_labels=name_target, cmap=plt.cm.Reds)
        RocCurveDisplay.from_predictions(
            target_test, clf_pro, ax=axes[1], name=name_clf)
        PrecisionRecallDisplay.from_predictions(
            target_test, clf_pro, ax=axes[2], name=name_clf)

    return clf_pro, clf_prediction, auc_roc, auc_pr

def clf_test_external_bootstrap(
    df_test,
    target_test,
    dataset_name,
    clfs,
    clf_names,
    preprocessor=None, 
    times_bootstrap=2000,
):
    """
    test the fitted model in external validation dataset with bootstrapping
    Input： df_test, clf, show_score_dis=0, show_not=0, col_target='tracheotomy', target_names=['Not','Tra']
    Output: the dataset shape and the performance (auroc, auprc)
    Paint: if set, the score distribution, confusion matrix, roc, prc
    Return: clf_pro, clf_prediction, auc_roc, auc_pr
    """

    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    df_result_clfs = pd.DataFrame(
        {
            "roc_mean": 0,
            "roc_CI": 0,
            "roc_std": 0,
            "pr_mean": 0,
            "pr_CI": 0,
            "pr_std": 0,
        },
        index=clf_names,
    )
    target_clfs, pro_test_clfs, roc_test_clfs, prc_test_clfs = [], [], [], []
    for clf, clf_name in zip(clfs, clf_names):
        target_clf, pro_test_clf, roc_test_clf, prc_test_clf = [], [], [], []
        for time in tqdm(range(times_bootstrap)):
            # seed = idx
            df_test_bootstrap, target_test_bootstrap = resample(
                df_test, target_test, stratify=target_test, random_state=time
            )
            pro_test, pred_test, roc_test, prc_test = clf_test_external(
                df_test_bootstrap, target_test=target_test_bootstrap, clf=clf, 
                preprocessor=preprocessor, name_clf=clf_name, show_not=1
            )
            target_clf.append(target_test_bootstrap)
            pro_test_clf.append(pro_test)
            roc_test_clf.append(roc_test)
            prc_test_clf.append(prc_test)
        # paint
        flat_target_clf = np.asarray(target_clf).ravel()
        flat_pro_test_clf = np.asarray(pro_test_clf).ravel()
        RocCurveDisplay.from_predictions(
            flat_target_clf, flat_pro_test_clf, ax=ax[0], name=clf_name
        )
        PrecisionRecallDisplay.from_predictions(
            flat_target_clf, flat_pro_test_clf, ax=ax[1], name=clf_name
        )
        mean_roc_clf, std_roc_clf, inter_roc_clf = cal_CI(
            roc_test_clf)
        mean_pr_clf, std_pr_clf, inter_pr_clf = cal_CI(
            prc_test_clf)
        df_result_clfs.loc[clf_name] = (
            mean_roc_clf,
            inter_roc_clf,
            std_roc_clf,
            mean_pr_clf,
            inter_pr_clf,
            std_pr_clf,
        )
        print(
            f"{clf_name}:The General AUC-ROC: {mean_roc_clf:.4f} ± {inter_roc_clf:.4f}, the General AP-PR:{mean_pr_clf:.4f} ± {inter_pr_clf:.4f}"
        )
        # save
        target_clfs.append(target_clf)
        pro_test_clfs.append(pro_test_clf)
        roc_test_clfs.append(roc_test_clf)
        prc_test_clfs.append(prc_test_clf)

    ax[0].set_title(f"ROC of {dataset_name}")
    ax[1].set_title(f"PRC of {dataset_name}")
    pyplot.tight_layout()
    if len(clfs) > 1:
        return df_result_clfs, target_clfs, pro_test_clfs, roc_test_clfs, prc_test_clfs
    else:
        return (
            df_result_clfs,
            target_clfs[0],
            pro_test_clfs[0],
            roc_test_clfs[0],
            prc_test_clfs[0],
        )


def clf_test_subgroups(clf, df_test_datasets, name_datasets, name_basis, col_label='tracheotomy', path_save=None):
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for df_test, [idx, name_dataset] in zip(df_test_datasets, enumerate(name_datasets)):
        print(f"{name_dataset}")
        pro_test, pred_test = clf_test_external(df_test, clf=clf, show_not=1)
        RocCurveDisplay.from_predictions(
            df_test[col_label], pro_test, ax=ax[0], color=palette(idx), name=name_dataset)
        PrecisionRecallDisplay.from_predictions(
            df_test[col_label], pro_test, ax=ax[1], color=palette(idx), name=name_dataset)

    # ax_1
    ax[0].plot([0, 1], [0, 1], linestyle='dashed', color='black')
    ax[0].set_xlabel("False Positive Rate", fontsize=15)
    ax[0].set_ylabel("True Positive Rate", fontsize=15)
    ax[0].set_title(f"ROC of {name_basis}", fontsize=15)
    ax[0].legend(loc="lower right", fontsize=15)
    # ax_2
    ax[1].plot([0, 1], [1, 0], linestyle='dashed', color='black')
    ax[1].set_xlabel("Recall", fontsize=15)
    ax[1].set_ylabel("Precision", fontsize=15)
    ax[1].set_title(f"PRC of {name_basis}", fontsize=15)
    ax[1].legend(loc="lower left", fontsize=15)

    pyplot.tight_layout()
    # Save
    if path_save:
        pyplot.savefig(path_save, format='svg', dpi=500,
                       bbox_inches='tight', pad_inches=0.1)


def clf_test_subgroups_bootstrap(clf, df_test_datasets, name_datasets, name_basis, col_label='tracheotomy', path_save=None, times_bootstrap=1000):
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    target_pps, pro_test_pps, roc_test_pps, prc_test_pps = [], [], [], []
    for df_test, [idx, name_dataset] in zip(df_test_datasets, enumerate(name_datasets)):
        print(f"{name_dataset}")
        target_clf, pro_test_clf, roc_test_clf, prc_test_clf = [], [], [], []
        for time in tqdm(range(times_bootstrap)):
            df_test_bootstrap = resample(
                df_test, stratify=df_test[col_label], random_state=time)
            target_test = df_test_bootstrap[col_label]
            pro_test, pred_test, roc_test, prc_test = clf_test_external(
                df_test_bootstrap, clf=clf, show_not=1)
            target_clf.append(target_test)
            pro_test_clf.append(pro_test)
            roc_test_clf.append(roc_test)
            prc_test_clf.append(prc_test)
        target_clf = np.asarray(target_clf).ravel()
        pro_test_clf = np.asarray(pro_test_clf).ravel()
        RocCurveDisplay.from_predictions(
            target_clf, pro_test_clf, ax=ax[0], color=palette(idx), name=name_dataset)
        PrecisionRecallDisplay.from_predictions(
            target_clf, pro_test_clf, ax=ax[1], color=palette(idx), name=name_dataset)
        mean_roc_clf, std_roc_clf, inter_roc_clf = cal_CI(roc_test_clf)
        mean_pr_clf, std_pr_clf, inter_pr_clf = cal_CI(prc_test_clf)
        print(f"{name_dataset}:The General AUC-ROC: {mean_roc_clf:.4f}±{inter_roc_clf:.4f}, the General AP-PR:{mean_pr_clf:.4f}±{inter_pr_clf:.4f}")
        # save
        target_pps.append(target_clf)
        pro_test_pps.append(pro_test_clf)
        roc_test_pps.append(roc_test_clf)
        prc_test_pps.append(prc_test_clf)

    # ax_1
    ax[0].plot([0, 1], [0, 1], linestyle='dashed', color='black')
    ax[0].set_xlabel("False Positive Rate", fontsize=15)
    ax[0].set_ylabel("True Positive Rate", fontsize=15)
    ax[0].set_title(f"ROC of {name_basis}", fontsize=15)
    ax[0].legend(loc="lower right", fontsize=15)
    # ax_2
    ax[1].plot([0, 1], [1, 0], linestyle='dashed', color='black')
    ax[1].set_xlabel("Recall", fontsize=15)
    ax[1].set_ylabel("Precision", fontsize=15)
    ax[1].set_title(f"PRC of {name_basis}", fontsize=15)
    ax[1].legend(loc="lower left", fontsize=15)

    pyplot.tight_layout()
    # Save
    if path_save:
        pyplot.savefig(path_save, format='svg', dpi=500,
                       bbox_inches='tight', pad_inches=0.1)

    return target_pps, pro_test_pps, roc_test_pps, prc_test_pps
