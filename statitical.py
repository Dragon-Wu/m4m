# import
from sklearn import datasets
from sklearn.metrics import brier_score_loss
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.calibration import CalibrationDisplay, calibration_curve
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
# paint
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib import rcParams
rcParams['font.size'] = 15  # 设置字体大小
palette = pyplot.get_cmap('tab10')


def calculate_net_benefit_model(y_pred_score, y_label, thresh_group):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score >= thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        # net_benefit = max(net_benefit, 0)
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model


def calculate_net_benefit_all(y_label, thresh_group):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all


def plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all, fill=1):
    # Plot
    # net_benefit_model = [ dd for dd in net_benefit_model if dd>0]
    # thresh_group = thresh_group[:len(net_benefit_model)]
    # net_benefit_all = thresh_group[:len(net_benefit_model)]
    ax.plot(thresh_group, net_benefit_model, color='crimson', label='Model')
    ax.plot(thresh_group, net_benefit_all, color='black', label='Treat all')
    ax.plot((0, 1), (0, 0), color='black', linestyle=':', label='Treat none')
    # Fill, Shows that the model is better than treat all and treat none The good part
    y2 = np.maximum(net_benefit_all, 0)
    y1 = np.maximum(net_benefit_model, y2)
    if fill:
        ax.fill_between(thresh_group, y1, y2, color='crimson', alpha=0.2)

    # Figure Configuration, Beautify the details
    ax.set_xlim(0, 1)
    # adjustify the y axis limitation
    y_min = min(net_benefit_model)
    y_max = max(net_benefit_model)
    y_more = (y_max - y_min)*0.1
    ax.set_ylim(-0.02, y_max)
    # ax.set_ylim(-0.05, y_max)
    ax.set_xlabel(
        xlabel='Threshold Probability',
    )
    ax.set_ylabel(
        ylabel='Net Benefit',
    )
    ax.grid('major')
    # ax.spines['right'].set_color((0.8, 0.8, 0.8))
    # ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc='upper right')
    return ax

# sample
# thresh_group = np.arange(0,1,0.01)
# targetS_DCA = targets_lr
# pros_DCA = probs_lr
# net_benefit_model = calculate_net_benefit_model(pros_DCA, targetS_DCA, thresh_group)
# net_benefit_all = calculate_net_benefit_all(targetS_DCA, thresh_group)
# fig, ax = plt.subplots(figsize=(8,6))
# ax = plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all)
# # fig.savefig('fig1.png', dpi = 300)
# plt.show()


def plot_calibration_curve(clf, name, ax, X_test, y_test, title):
    y_pred = clf.predict(X_test)
    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(X_test)[:, 1]
    else:  # use decision function
        prob_pos = clf.decision_function(X_test)
        prob_pos = \
            (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

    clf_score = brier_score_loss(y_test, prob_pos, pos_label=y_test.max())

    fraction_of_positives, mean_predicted_value = \
        calibration_curve(y_test, prob_pos, n_bins=10, normalize=False)

    ax.plot(mean_predicted_value, fraction_of_positives, "s-",
            label="%s (%1.3f)" % (name, clf_score), alpha=0.5, color='k')

    ax.set_ylabel("Fraction of positives")
    ax.set_ylim([-0.05, 1.05])
    ax.set_title(title)

    ax.set_xlabel("Mean predicted value")

    plt.tight_layout()
    return clf_score

# sample
# fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(6, 12))

# ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated",)
# ax2.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
# ax3.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

# scores = {'Method 1': [], 'Method 2': [], 'Method 3': []}

# for i in range(0, 100):

#     X, y = datasets.make_classification(n_samples=10000, n_features=200,
#                                         n_informative=10, n_redundant=10,
#                                         # random_state=42,
#                                         n_clusters_per_class=1, weights=[0.8, 0.2])

#     X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.80,
#                                                                 # random_state=42
#                                                                 )

#     X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.80,
#                                                       # random_state=42
#                                                       )

#     #my_clf = GaussianNB()
#     my_clf = LogisticRegression(max_iter=10000)

#     # Method 1, train classifier within CCCV
#     model = CalibratedClassifierCV(my_clf)
#     model.fit(X_train_val, y_train_val, n_jobs-1)
#     r = plot_calibration_curve(
#         model, "all_cal", ax1, X_test, y_test, "Method 1")
#     scores['Method 1'].append(r)

#     # Method 2, train classifier and then use CCCV on DISJOINT set
#     my_clf.fit(X_train, y_train)
#     model = CalibratedClassifierCV(my_clf, cv='prefit')
#     model.fit(X_val, y_val, n_jobs-1)
#     r = plot_calibration_curve(
#         model, "all_cal", ax2, X_test, y_test, "Method 2")
#     scores['Method 2'].append(r)

#     # Method 3, train classifier on set, then use CCCV on SAME set used for training
#     my_clf.fit(X_train_val, y_train_val)
#     model = CalibratedClassifierCV(my_clf, cv='prefit')
#     model.fit(X_train_val, y_train_val, n_jobs-1)
#     r = plot_calibration_curve(
#         model, "all_cal", ax3, X_test, y_test, "Method 2 non Dis")
#     scores['Method 3'].append(r)

# b = pd.DataFrame(scores).boxplot()
# plt.suptitle('Brier score')
