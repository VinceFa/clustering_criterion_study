from __future__ import division

import sklearn
import sklearn.datasets
import sklearn.cluster
import numpy as np
import scipy as sc
import converted_metrics
import new_metrics
import utility
import main
import random
import matplotlib.pyplot as plt

# Some tests to discover sklearn, check metrics implemented, etc...

def tests_perso():
    iris = sklearn.datasets.load_iris()
    # print(iris)
    print(iris.keys())
    D = iris['data']
    var = sklearn.cluster.KMeans(4)
    var.fit(D)
    # print(var)
    pred = var.predict(D)
    # print(pred)
    # print(iris)
    # print(precis)

    print("IRIS")
    print(pred[1:50])
    print(pred[51:100])
    print(pred[101:150])
    print(goodness(iris['target'], pred))
    print(goodness(iris['target'], pred, 'macro'))
    print(goodness(iris['target'], pred, 'micro'))
    print(goodness(iris['target'], pred, 'weighted'))
    print("")
    print("WINE")

    wine = sklearn.datasets.load_wine()
    var2 = sklearn.cluster.KMeans(4)
    var2.fit(wine['data'])
    pred2 = var2.predict(wine['data'])
    print(goodness(wine['target'], pred2))
    print(goodness(wine['target'], pred2, "macro"))
    print(goodness(wine['target'], pred2, "micro"))
    print(goodness(wine['target'], pred2, "weighted"))
    print("Contigence :")
    cont = sklearn.metrics.cluster.contingency_matrix(wine['target'], pred2)
    print(cont)
    sum_comb_c = np.ravel(cont.sum(axis=0))
    print(sum_comb_c)

def fm_comparison(y_true, y_pred):
    # Comparison of Fowlkes & Mallow values from sklearn module and from personnal implementation
    contingency = compute_contingency(y_true, y_pred)
    calc_fm = tp / np.sqrt((tp + fp) * (tp + fn))
    real_fm = sklearn.metrics.fowlkes_mallows_score(y_true, y_pred)
    print("(manually calculated FM, sklearn FM) = ", calc_fm, real_fm)

def test_accuracy():
    wine = sklearn.datasets.load_wine()
    var2 = sklearn.cluster.KMeans(4)
    var2.fit(wine['data'])
    pred2 = var2.predict(wine['data'])

    converted_metrics.accuracy(wine['target'], pred2)


# tests_perso()
# test_accuracy()

def test_entropy():
    wine = sklearn.datasets.load_wine()
    var2 = sklearn.cluster.KMeans(4)
    var2.fit(wine['data'])
    pred2 = var2.predict(wine['data'])

    contingency = utility.compute_contingency(wine['target'], pred2)

    entropy = new_metrics.cond_entropy(wine['target'], pred2, contingency)

    print("true", wine['target'])
    print("predicted", pred2)
    print("entropy", entropy)


def test_purity():
    wine = sklearn.datasets.load_wine()
    var2 = sklearn.cluster.KMeans(4)
    var2.fit(wine['data'])
    pred2 = var2.predict(wine['data'])

    purity = new_metrics.purity(wine['target'], pred2)
    i_p = new_metrics.inversed_purity(wine['target'], pred2)

    print("true", wine['target'])
    print("predicted", pred2)
    print("purity", purity)

    print("i_p", i_p)


def test_mutual_information_cond_entropy():
    wine = sklearn.datasets.load_wine()
    var2 = sklearn.cluster.KMeans(4)
    var2.fit(wine['data'])
    pred2 = var2.predict(wine['data'])
    cont = utility.compute_contingency(wine['target'], pred2)
    print("cont", cont)

    # Implementation perso
    CE = new_metrics.cond_entropy(wine['target'], pred2)
    print("found CE : ", CE)

    # Calcul depuis CE = H(label) - MI
    MI = sklearn.metrics.cluster.mutual_info_score(wine['target'], pred2)
    H_label = sklearn.metrics.cluster.entropy(wine['target'])
    sub = H_label - MI
    print("CI from H_label - MI : ", sub)

    # Calcul depuis hom = 1 - [CE / H(label)]
    H_cluster = sklearn.metrics.cluster.entropy(wine['target'])
    homogeneity = sklearn.metrics.cluster.homogeneity_score(wine['target'], pred2)
    ce_complet = H_label * (1 - homogeneity)
    print("CI from homogeneity : ", ce_complet)


def test_ari():
    # Comparison of ARI values from sklearn module and from personnal implementation
    # Note : Currently not using TP/FP/FN/TN values ; Should change it soon
    contingency = compute_contingency(y_true, y_pred)
    ri = np.sum(comb(contingency,
                     2))  # note : il ne s'agit pas réellement de ri, mais ri après simplifcation de la formule d'ari
    eri = (np.sum(comb(a_i, 2)) * np.sum(comb(b_j, 2))) / comb(N, 2)
    maxri = 0.5 * (np.sum(comb(a_i, 2)) + np.sum(comb(b_j, 2)))
    calc_ari = converted_metrics.adjusted_rand_index(y_true, y_pred, contingency)

    real_ari = sklearn.metrics.adjusted_rand_score(y_true, y_pred)

    print("(manually calculated ARI, sklearn ARI) = ", calc_ari, real_ari)


def test_get_metrics():
    dataset = sklearn.datasets.load_breast_cancer()
    var2 = sklearn.cluster.KMeans(4)
    var2.fit(dataset['data'])
    true = dataset['target']
    pred = var2.predict(dataset['data'])
    print("H(reel)", sklearn.metrics.cluster.entropy(true))
    print("H(pred)", sklearn.metrics.cluster.entropy(pred))
    MI = sklearn.metrics.cluster.mutual_info_score(true, pred)
    print("MI", MI)
    cont = utility.compute_contingency(true, pred)
    rep = main.get_metrics(true, pred)
    for a in rep:
        print(a, "\t", rep[a])

    print(cont)


def test_info():
    dataset = sklearn.datasets.load_iris()
    print(type(dataset))
    print(dataset['DESCR'])
    print(dataset['data'])
    var2 = sklearn.cluster.KMeans(4)
    var2.fit(dataset['data'])
    true = dataset['target']

def test_MS():
    for i in range(100):
        true = random.randint(1,1000)
        print("TRUE:"+str(true))
        predit = main.get_meanshift_prediction(None,true)
        print("PREDIT:"+str(predit))

def dbscan():
    fi = "C:/Users/Vincent/Desktop/Stage PFE/Phase 3 - Tests sur datasets/Datasets propres/breast_tissue_features"
    file = open(fi,"r")

    data_values = []
    for line in file:
        data_values = data_values + [[float(num) for num in line.split(',')]]
    data_values = np.array(data_values)
    file.close()
    print(np.shape(data_values))
    EPS = []
    NB = []
    for eps in range(1,10000,):
        dbscan = sklearn.cluster.DBSCAN(eps=(eps*0.01)).fit(data_values)
        prediction = dbscan.labels_
        cluster_number = len(np.unique(prediction))
        print(cluster_number)
        EPS = EPS + [eps*0.01]
        NB = NB + [cluster_number]

    plt.plot(EPS,NB,'*')
    plt.show()



dbscan()

# FIRST TEST WITH BREAST CANCER DATASET

"""
mi 	 5.964165305284491              FAUX : BCP TROP GRAND (facteur N manquant)
ari 	 0.6521521240094236         FAUSSE PAR MI
ami 	 6.894341222493518          FAUSSE PAR MI
compl 	 5.234758469029095          FAUSSE PAR MI
homog 	 9.032284771615036          FAUSSE PAR MI
vmeasure 	 6.628118861825169      FAUSSE PAR MI
entropy 	 0.2806012203533871     VALIDE
accuracy 	 0.7009641327755638     VALIDE
precision 	 0.8280647244671752     VALIDE
recall 	 0.8280647244671752         ERREUR DETECTEE : FAUSSE FORMULE TP/(TP+FN)
falsealarm 	 0.20383668232518676    VALIDE
fm 	 70155.63104983092              FAUX : BCP TROP GRAND
F1 	 0.8280647244671752             FAUSSE PAR LE RECALL FAUX
purity 	 0.8347978910369068         VALIDE
inv_purity 	 0.6362038664323374     VALIDE
epratio 	 0.33613072501279423    VALIDE
jaccard 	 0.49538439046803534    VALIDE
nmi 	 6.876178386494516          VALIDE => FAUSSE PAR MI
ri 	 0.7009641327755638             VALIDE
vi 	 24.975368402292748             ERREUR DETECTEE : FACTEUR nij/N MANQUANT
goodness 	 0.8280647244671752     ERREUR DETECTEE : FAUSSE PAR RECALL
balacc 	 0.7296104154802536         VALIDE
q2 	 0.7801497877135831
[[  6. 262.]
 [100.   1.]
 [ 87.  94.]
 [ 19.   0.]]
"""

# SECOND TEST WITH WINE DATASET

"""
mi 	 0.4570864142354184             VALIDE
ari 	 0.30289726650864374        VALIDE
ami 	 0.3716144449592707         NON VERIFIABLE : EMI non relevé
compl 	 0.3442650232242263         VALIDE
homog 	 0.4208749855144595         VALIDE
vmeasure 	 0.37873470216642585    VALIDE
entropy 	 0.6289520294052646     VALIDE
accuracy 	 0.702723290801752      VALIDE
precision 	 0.574517554057196      VALIDE
recall 	 0.4641247182569497         VALIDE
falsealarm 	 0.39077514413837283    VALIDE
fm 	 0.5163795095765012             VALIDE
f1 	 0.5134545454545454             VALIDE
purity 	 0.7247191011235955         VALIDE
inv_purity 	 0.5955056179775281     VALIDE
epratio 	 0.8678562886367217     VALIDE
jaccard 	 0.34540117416829746    VALIDE
nmi 	 0.3806475228641733         VALIDE
ri 	 0.702723290801752              VALIDE
vi 	 1.4995822967965748             VALIDE
goodness 	 0.5193211361570729     VALIDE
balacc 	 0.6626953819884303         VALIDE
q2 	 0.6811246527626528
[[ 0. 45. 12.]
 [23.  0.  0.]
 [ 6. 22. 31.]
 [30.  4.  5.]]
 """

# TROISIEME TEST AVEC DIGITS

"""
mi 	 0.8498384891687666             VALIDE
ari 	 0.2952514475560513         A RETIRER (VALIDE D OFFICE)
ami 	 0.47395981809298066        VALIDE
compl 	 0.6143446333224005         VALIDE
homog 	 0.3690971373072919         VALIDE
vmeasure 	 0.46114137562858926    VALIDE
entropy 	 1.4526407317991095     VALIDE
accuracy 	 0.7881553393245114     VALIDE
precision 	 0.27628477133427626    VALIDE
recall 	 0.6969413933099206         VALIDE
falsealarm 	 0.8576296968012754     VALIDE
fm 	 0.4388100881748542             VALIDE
f1 	 0.39570307597395815            VALIDE
purity 	 0.3656093489148581         VALIDE
inv_purity 	 0.7851975514746801     VALIDE
epratio 	 3.9732045586651443     VALIDE
jaccard 	 0.24665201936617       VALIDE
nmi 	 0.47618572582490953        VALIDE
ri 	 0.7881553393245114             VALIDE
vi 	 1.9861275670053695             VALIDE
goodness 	 0.48661308232209843    VALIDE
balacc 	 0.6180074355804672         VALIDE
q2 	 0.6927388563475838             VALIDE
[[  0. 110. 154.  12.   3.  85.   1.  22.  95.   3.]
 [  0.  58.   9.   4. 142.  12.   1. 157.  30.  30.]
 [175.   0.   1.   0.  36.   4. 179.   0.   1.   0.]
 [  3.  14.  13. 167.   0.  81.   0.   0.  48. 147.]]"""
