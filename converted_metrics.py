from __future__ import division

import sklearn
import numpy as np
from scipy.special import comb
import utility

from scipy import sparse as sp
from sklearn.metrics.cluster.expected_mutual_info_fast import expected_mutual_information

# All formulas are presented in the report
# Each time it is possible, all intermediate values can be give as an input

# This file contains 3 types of functions :
# All metrics that are already implemented in sklearn but concerned by the "absolute labelling" problem presented in the report
# All the other metrics (absent in the sklearn module) that also needed to be adapted because of this problem.
# Reimplementation of other functions (already present in sklearn), that are more efficient for large-scale benchmarking


# REIMPLEMENTATION OF ALREADY EXISTING CRITERIA

def jaccard_index(y_true, y_pred, contingency=None, PBV=None):
    if contingency is None:
        contingency = utility.compute_contingency(y_true, y_pred)
    if PBV is None:
        PBV = utility.pair_based_values(contingency)
    tp, fp, fn, tn = PBV
    return tp / (tp + fp + fn)


def precision(y_true, y_pred, contingency=None, PBV=None):
    if contingency is None:
        contingency = utility.compute_contingency(y_true, y_pred)
    if PBV is None:
        PBV = utility.pair_based_values(contingency)
    tp, fp, _, _ = PBV
    return tp / (tp + fp)


def accuracy(y_true, y_pred, contingency=None, PBV=None):
    # Note : Identical to Rand Index criterion, in context of pair counting
    if contingency is None:
        contingency = utility.compute_contingency(y_true, y_pred)
    if PBV is None:
        PBV = utility.pair_based_values(contingency)
    tp, fp, fn, tn = PBV
    return (tp + tn) / (tp + tn + fp + fn)


def f_beta_score(y_true, y_pred, beta, contingency=None, PBV=None, pre=None, rec=None):
    if contingency is None:
        contingency = utility.compute_contingency(y_true, y_pred)
    if PBV is None:
        PBV = utility.pair_based_values(contingency)
    if pre is None:
        pre = precision(y_true, y_pred, contingency, PBV)
    if rec is None:
        rec = recall(y_true, y_pred, contingency, PBV)
    beta2 = np.power(beta, 2)
    if pre * rec == 0:
        return 0
    return ((1 + beta2) * pre * rec) / (beta2 * pre + rec)


def recall(y_true, y_pred, contingency=None, PBV=None):
    if contingency is None:
        contingency = utility.compute_contingency(y_true, y_pred)
    if PBV is None:
        PBV = utility.pair_based_values(contingency)
    tp, _, fn, _ = PBV
    return tp / (tp + fn)


# IMPLEMENTATION OF CRITERIA CONCERNED BY THE ABSOLUTE LABELING PROBLEM

def balanced_accuracy(y_true, y_pred, contingency=None, PBV=None):
    if contingency is None:
        contingency = utility.compute_contingency(y_true, y_pred)
    if PBV is None:
        PBV = utility.pair_based_values(contingency)
    tp, fp, fn, tn = PBV
    a = tp / (tp + fn)
    b = tn / (tn + fp)
    return 0.5 * a + 0.5 * b


def clustering_error(y_true, y_pred, contingency=None, PBV=None):
    # Note : It might be different to consider CE and Accuracy, especially considering subtle variation like micro/macro/weighted averaging.
    # However, in data pairs context, CE is always equal to 1 - Accuracy
    if contingency is None:
        contingency = utility.compute_contingency(y_true, y_pred)
    if PBV is None:
        PBV = utility.pair_based_values(contingency)
    print(
        "'Clustering Error' is redundant criterion ; Please use Accuracy criterion instead, and compute  1 - Accuracy")


def false_alarm_rate(y_true, y_pred, contingency=None, PBV=None):
    if contingency is None:
        contingency = utility.compute_contingency(y_true, y_pred)
    if PBV == None:
        PBV = utility.pair_based_values(contingency)
    _, fp, fn, _ = PBV
    if fp + fn == 0:
        return 0
    return fp / (fp + fn)


def goodness(y_true, y_pred, contingency=None, PBV=None, pre=None, rec=None):
    if contingency is None:
        contingency = utility.compute_contingency(y_true, y_pred)
    if PBV is None:
        PBV = utility.pair_based_values(contingency)
    if pre is None:
        pre = precision(y_true, y_pred, contingency, PBV)
    if rec is None:
        rec = recall(y_true, y_pred, contingency, PBV)
    return 0.5 * (pre + rec)


# REIMPLEMENTATION OF FUNCTIONS THAT ARE MORE SUITABLE FOR LARGE-SCALE BENCHMARKING

def homogeneity(y_true, y_pred, contingency=None, mutual_information=None, real_clustering_entropy=None):
    if contingency is None:
        contingency = utility.compute_contingency(y_true, y_pred)
    if mutual_information is None:
        mutual_information = sklearn.metrics.cluster.mutual_info_score(y_true, y_pred, contingency)
    if real_clustering_entropy is None:
        real_clustering_entropy = sklearn.metrics.cluster.entropy(y_true)
    return mutual_information / real_clustering_entropy


def completness(y_true, y_pred, contingency=None, mutual_information=None, predicted_clustering_entropy=None):
    if contingency is None:
        contingency = utility.compute_contingency(y_true, y_pred)
    if mutual_information is None:
        mutual_information = sklearn.metrics.cluster.mutual_info_score((y_true, y_pred, contingency))
    if predicted_clustering_entropy is None:
        predicted_clustering_entropy = sklearn.metrics.cluster.entropy(y_pred)
    return mutual_information / predicted_clustering_entropy


def v_measure(y_true, y_pred, beta, mutual_information=None, contingency=None, predicted_clustering_entropy=None,
              real_clustering_entropy=None, homog=None, compl=None):
    # Enhancement that include a beta parameter ; Optionnal, since no paper used the v-measure criterion with beta != 1
    if contingency is None:
        contingency = utility.compute_contingency(y_true, y_pred)
    if mutual_information is None:
        mutual_information = sklearn.metrics.mutual_info_score(y_true, y_pred, contingency)
    if predicted_clustering_entropy is None:
        predicted_clustering_entropy = sklearn.metrics.entropy(y_pred)
    if real_clustering_entropy is None:
        real_clustering_entropy = sklearn.metrics.entropy(y_true)
    if homog is None:
        homog = homogeneity(y_true, y_pred, contingency, mutual_information, real_clustering_entropy)
    if compl is None:
        compl = completness(y_true, y_pred, contingency, mutual_information, predicted_clustering_entropy)
    if homog + compl == 0:
        return 0
    return ((1 + beta) * homog * compl) / (beta * homog + compl)


def adjusted_rand_index(y_true, y_pred, contingency=None, PBV=None, a_i=None, b_j=None):
    if contingency is None:
        contingency = compute_contingency(y_true, y_pred)
    if PBV is None:
        PBV = utility.pair_based_values(contingency)
    if a_i is None:
        a_i = utility.compute_a_i(contingency)
    if b_j is None:
        b_j = utility.compute_b_j(contingency)
    tp, fp, fn, tn = PBV
    N = np.sum(a_i)
    ri = np.sum(comb(contingency, 2))  # note : not truly ri, missing division by comb(N,2)
    eri = (np.sum(comb(a_i, 2)) * np.sum(comb(b_j, 2))) / comb(N, 2)  # idem
    maxri = 0.5 * (np.sum(comb(a_i, 2)) + np.sum(comb(b_j, 2)))  # idem
    return (ri - eri) / (maxri - eri)


def mutual_information(y_true, y_pred, contingency=None, a_i=None, b_j=None):
    if contingency is None:
        contingency = utility.compute_contingency(y_true, y_pred)
    if a_i is None:
        a_i = utility.compute_a_i(contingency)
    if b_j is None:
        b_j = utility.compute_b_j
    J_a_i = np.repeat(a_i, np.ma.size(b_j), axis=1)
    I_b_j = np.repeat(b_j, np.ma.size(a_i), axis=0)
    N = np.sum(a_i)

    tmp = np.true_divide(N * contingency, J_a_i * I_b_j)
    log_value = np.ma.log(tmp)
    log_value = log_value.filled(0)

    return np.sum(contingency * log_value) / N


def normalized_mutual_information(y_true, y_pred, contingency=None, a_i=None, b_j=None, mi=None,
                                  real_clustering_entropy=None, predicted_clustering_entropy=None):
    if contingency is None:
        contingency = utility.compute_contingency(y_true, y_pred)
    if a_i is None:
        a_i = utility.compute_a_i(contingency)
    if b_j is None:
        b_j = utility.compute_b_j
    if mi is None:
        mi = mutual_information(y_true, y_pred, contingency, a_i, b_j)
    if real_clustering_entropy is None:
        real_clustering_entropy = sklearn.metrics.cluster.entropy(y_true)
    if predicted_clustering_entropy is None:
        predicted_clustering_entropy = sklearn.metrics.cluster.entropy(y_pred)
    return mi / np.sqrt(real_clustering_entropy * predicted_clustering_entropy)


def adjusted_mutual_information(y_true, y_pred, contingency=None, a_i=None, b_j=None, mi=None,
                                real_clustering_entropy=None, predicted_clustering_entropy=None):
    if contingency is None:
        contingency = utility.compute_contingency(y_true, y_pred)
    if a_i is None:
        a_i = utility.compute_a_i(contingency)
    if b_j is None:
        b_j = utility.compute_b_j
    if mi is None:
        mi = mutual_information(y_true, y_pred, contingency, a_i, b_j)
    if real_clustering_entropy is None:
        real_clustering_entropy = sklearn.metrics.cluster.entropy(y_true)
    if predicted_clustering_entropy is None:
        predicted_clustering_entropy = sklearn.metrics.cluster.entropy(y_pred)
    J_a_i = np.repeat(a_i, np.ma.size(b_j), axis=1)
    I_b_j = np.repeat(b_j, np.ma.size(a_i), axis=0)

    N = np.sum(contingency)

    emi = expected_mutual_information(contingency, N)
    maxmi = np.sqrt(real_clustering_entropy * predicted_clustering_entropy)

    return (mi - emi) / (maxmi - emi)


def fowlkes_mallows(y_true, y_pred, contingency=None, PBV=None):
    if contingency is None:
        contingency = utility.compute_contingency(y_true, y_pred)
    if PBV is None:
        PBV = utility.pair_based_values(contingency)
    tp, fp, fn, tn = PBV
    return np.sqrt((tp * tp) / ((tp + fp) * (tp + fn)))
