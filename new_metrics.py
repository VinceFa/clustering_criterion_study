from __future__ import division

import sklearn
import sklearn.datasets
import sklearn.cluster
import converted_metrics
import utility
import numpy as np
from scipy.special import comb

# This file contains all criterion that aren't present in sklearn module
# All formulas are presented in the report
# Each time it is possible, all intermediate values can be give as an input

def cond_entropy(y_true, y_pred, contingency=None, a_i=None):
    # Compute the average of each predicted-cluster's entropy, which is based on membership diversity of each data on reals clusters
    # Must be distinguished from entropy
    if contingency is None:
        contingency = utility.compute_contingency(y_true, y_pred)
    if a_i is None:
        a_i = utility.compute_a_i(contingency)

    J_a_i = np.repeat(a_i, np.size(contingency, axis=1), axis=1)
    log_value = np.true_divide(contingency, J_a_i)
    log_value = np.ma.log(log_value)  # Required to deal with 0's
    log_value = log_value.filled(0)
    N = np.ma.sum(a_i)
    return - np.sum(contingency * log_value) / N


def inversed_purity(y_true, y_pred, contingency=None):
    # Note : Simply the definition of Purity, where the predicted and the real clusters roles are inverted
    if contingency is None:
        contingency = utility.compute_contingency(y_true, y_pred)
    return purity(y_pred, y_true, np.transpose(contingency))


def purity(y_true, y_pred, contingency=None):
    if contingency is None:
        contingency = utility.compute_contingency(y_true, y_pred)
    max_i = np.max(contingency, axis=1)
    return np.sum(max_i) / np.ma.size(y_true)


def q2(y_true, y_pred, contingency=None, cond_entrop=None, a_i=None, b_j=None):
    if contingency is None:
        contingency = utility.compute_contingency(y_true, y_pred)
    if cond_entrop is None:
        cond_entrop = cond_entropy(y_true, y_pred, contingency)
    if a_i is None:
        a_i = utility.compute_a_i(contingency)
    if b_j is None:
        b_j = utility.compute_b_j(contingency)

    J = np.ma.size(b_j)
    N = np.sum(a_i)

    combi = comb(a_i + J - 1, J - 1)
    log = np.ma.log(combi)
    log = log.filled(0)

    q0 = cond_entrop + (np.sum(log) / N)

    logmin = comb(b_j + J - 1, J - 1)
    logmin = np.ma.log(logmin)
    logmin = logmin.filled(0)

    maxq0 = sklearn.metrics.cluster.entropy(y_true) + np.log(J)
    minq0 = np.sum(logmin) / N

    return (maxq0 - q0) / (maxq0 - minq0)


def rand_index(y_true, y_pred, contingency=None, PBV=None):
    if contingency is None:
        contingency = utility.compute_contingency(y_true, y_pred)
    if PBV is None:
        PBV = utility.pair_based_values(contingency)
    tp, fp, fn, tn = PBV
    return (tp + tn) / (tp + fp + fn + tn)


def variation_of_information(y_true, y_pred, contingency=None, a_i=None, b_j=None):
    if contingency is None:
        contingency = utility.compute_contingency(y_true, y_pred)
    if a_i is None:
        a_i = utility.compute_a_i(contingency)
    if b_j is None:
        b_j = utility.compute_b_j(contingency)
    J_a_i = np.repeat(a_i, np.ma.size(b_j), axis=1)
    I_b_j = np.repeat(b_j, np.ma.size(a_i), axis=0)
    N = np.sum(a_i)

    log1 = np.true_divide(contingency, J_a_i)
    log1 = np.ma.log(log1)
    log1 = log1.filled(0)
    log2 = np.true_divide(contingency, I_b_j)
    log2 = np.ma.log(log2)
    log2 = log2.filled(0)

    return - np.sum(contingency * (log1 + log2)) / N


def ep_ratio(y_true, y_pred, contingency=None, a_i=None, entr=None, pur=None):
    if contingency is None:
        contingency = utility.compute_contingency(y_true, y_pred)
    if entr is None:
        entr = cond_entropy(y_true, y_pred, contingency, a_i)
    if pur is None:
        pur = purity(y_true, y_pred, contingency)
    return entr / pur
