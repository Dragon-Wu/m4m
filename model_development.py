import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
from m4m.metrics import (
    get_general_auc,
    paint_general_cm,
    paint_general_roc_pr,
    paint_prob_distribution,
)
from m4m.simpletools import ListArray_2_FlattenArray

# model
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool


def clf_train_test_cv(
    data,
    target,
    cv,
    clf,
    preprocessor=None,
    target_names=["Tra", "Not"],
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
        if show_each:
            print(
                classification_report(
                    target_test, clf_prediction, target_names=target_names
                )
            )
            print(
                "-------------------------------------------------------------------------------"
            )
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
        paint_general_cm(list_target_clfs, list_pred_clfs)
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


# from sklearn.naive_bayes import GaussianNB
# clf_base = GaussianNB()
# default
# best_, index_, datas_, targets_, pros_, predictions_ = clf_train_test_cv(data, target, cv_stratified, clf_base, show_each=0, show_cm=1, show_roc_pr=1, show_score_dis=1, show_plotly=0)
# for completed results
# best_, index_, datas_, targets_, pros_, predictions_ = clf_train_test_cv(data, target, cv_stratified, clf_base, show_all=1)
# for simple results
# best_, index_, datas_, targets_, pros_, predictions_ = clf_train_test_cv(data, target, cv_stratified, clf_base, show_not=1)
