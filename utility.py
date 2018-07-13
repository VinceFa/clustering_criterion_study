from __future__ import division

import sklearn
import numpy as np
import os

from scipy.special import comb


def compute_a_i(contingency):
    # Return a 1xI matrix corresponding to the marginal total of data points in each of the I predicted clusters
    rep = np.sum(contingency, axis=1)
    return np.reshape(rep, (np.ma.size(rep), 1))


def compute_b_j(contingency):
    # Return a Jx1 matrix corresponding to the marginal total of data points in each of the J true clusters
    rep = np.sum(contingency, axis=0)
    return np.reshape(rep, (1, np.ma.size(rep)))


def pair_based_values(contingency):
    # Return the quadruplet (SS, SD, DS, DD) corresponding to the 4 categories of data pairs
    # For conveniency (since all the criteria originally use the TP, FP, FN, TN values), the name of these variables
    # in pair_based_values() and functions that use results of pair_based_values() will be tp, fp, fn, tn
    # (correspondance : tp <-> SS, fp <-> SD, fn <-> DS, tn <-> DD)
    a_i = compute_a_i(contingency)
    b_j = compute_b_j(contingency)
    N = sum(a_i)

    tp = np.sum(comb(contingency, 2))
    J_a_i = np.repeat(a_i, np.ma.size(b_j), axis=1)
    fp = np.sum(np.multiply(J_a_i - contingency, contingency))
    fp = 0.5 * fp
    I_b_j = np.repeat(b_j, np.ma.size(a_i), axis=0)
    fn = np.sum(np.multiply(I_b_j - contingency, contingency))
    fn = 0.5 * fn

    tn = np.sum(np.multiply(contingency, N - J_a_i - I_b_j + contingency))
    tn = 0.5 * tn

    return tp, fp, fn, tn


def compute_contingency(y_true, y_pred):
    # Custom calculation of contingency table, since adopted notation require IxJ contingency matrix and sklearn contingency
    # function produce JxI matrix
    tmp = np.transpose(sklearn.metrics.cluster.contingency_matrix(y_true, y_pred, sparse=False))
    return np.array(tmp, dtype='float64')  # Cast to avoid overflow problems

def pair_sum_test(y_true, y_pred):
    # Simple test to check if the sum of pairs tp,fp,fn,tn from pair_based_values() is equal to
    # total possible pair :  N comb 2
    contingency = compute_contingency(y_true, y_pred)
    tp, fp, fn, tn = pair_based_values(contingency)
    N = np.ma.size(y_true)
    pair_sum = tp + fp + fn + tn
    real_pair_number = comb(N, 2)
    print("(TP+FP+FN+TN, comb(N,2) = ", pair_sum, real_pair_number)


def normalize(matrix):
    # Input : Matrix of values (integer/float only), without header/footer/etc, and return a normalized matrix (by
    # feature, ie by column), using Z normalization
    instance_nb = np.size(matrix, axis=0)

    mean = np.mean(matrix, axis=0)
    mean = np.expand_dims(mean, 0)
    mean = np.repeat(mean, instance_nb, axis=0)

    std = np.std(matrix, axis=0)
    std = np.expand_dims(std, 0)
    std = np.repeat(std, instance_nb, axis=0)
    normalized_matrix = (matrix - mean) / std # Z normalization
    normalized_matrix = np.nan_to_num(normalized_matrix)
    return normalized_matrix


def formate():
    # Note : Function that need to be customized for each dataset that needs to be cleaned

    common_directory_original = "C:/Users/Vincent/Desktop/Stage PFE/Phase 3 - Tests sur datasets/Datasets bruts de l'UCI/"
    clean_data_directory = "C:/Users/Vincent/Desktop/Stage PFE/Phase 3 - Tests sur datasets/Datasets propres/"

    sub_directory = "fertility"
    filename = "fertility_Diagnosis.txt"


    file = common_directory_original + sub_directory + "/" + filename
    f = open(file, 'r')

    matrix = []
    header = False
    for line in f:
        if header:
            header = False
        else:
            if line.find("?") == -1 and line.find("NaN") == -1 and line.find(";;") == -1 :
                # line = line.replace("\n","")
                 # str = line[-13:]
                 # print(str)
                 # if str == "1\t0\t0\t0\t0\t0\t0" :
                 #     line = line[0:-13]  + "0"
                 # if str == "0\t1\t0\t0\t0\t0\t0" :
                 #     line = line[0:-13]  + "1"
                 # if str == "0\t0\t1\t0\t0\t0\t0" :
                 #     line = line[0:-13]  + "2"
                 # if str == "0\t0\t0\t1\t0\t0\t0" :
                 #     line = line[0:-13]  + "3"
                 # if str == "0\t0\t0\t0\t1\t0\t0" :
                 #     line = line[0:-13]  + "4"
                 # if str == "0\t0\t0\t0\t0\t1\t0" :
                 #     line = line[0:-13]  + "5"
                 # if str == "0\t0\t0\t0\t0\t0\t1" :
                 #     line = line[0:-13]  + "6"

                line = line.replace("O","0")
                line = line.replace("N","1")

                matrix = matrix + [[float(num) for num in line.split(',')]]

    nump = np.array(matrix)
    f.close()
    features = nump[:,:-1]
    featuresN = normalize(features)
    classes = np.array(nump[:,-1])

    if os.path.isfile(common_directory_original + sub_directory + "/" + "features"):
        warning = input("Ecraser ? (y/n)")
        if warning == "n":
            return
    np.savetxt(common_directory_original + sub_directory + "/" + "features", featuresN, delimiter=',')
    np.savetxt(common_directory_original + sub_directory + "/" + "classes", classes, delimiter=',', fmt="%i")

    np.savetxt(clean_data_directory + sub_directory +"_features", featuresN, delimiter=',')
    np.savetxt(clean_data_directory + sub_directory + "_classes", classes, delimiter=',', fmt="%i")

def load_csv(path):
    # Given a file in csv format (seperator : ',') , return a file
    file = open(path,'r')
    matrix = []
    for line in file:
        matrix = matrix + [[float(num) for num in line.split(',')]]
    file.close()
    return matrix

def save_csv(path,data):
    # Save a matrix into a csv file
    np.savetxt(path, data, delimiter=',',fmt="%i")