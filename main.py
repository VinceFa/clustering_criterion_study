from __future__ import division

import sklearn
import sklearn.datasets
import sklearn.cluster
import numpy as np
import scipy as sc
import converted_metrics
import new_metrics
import utility
import os
import time
import random
import copy
import matplotlib.pyplot as plt

# Algorithm and criterion to consider for the experiments
# Note : Due to implementation, if algorithm or criterion is REMOVED in the middle of process, dynamic programming doesn't work anymore, and each step has to be computed again (folders and files concerned need to be emptied)
ALGORITHM_LIST = {"kmean", "meanshift", "complete_agglo", "ward_agglo", "dbscan"}
CRITERION_LIST = {"mi", "ari", "ami", "compl", "homog", "vmeasure", "entropy", "precision", "recall", "falsealarm", "fm", "f1", "purity", "inv_purity", "epratio", "jaccard", "nmi", "ri", "vi", "goodness", "balacc", "q2"}

# Necessary distinction to establish algorithm ranking for each scenario
MORE_IS_BETTER = {'mi','ami','precision','recall','ri','ari','f1','purity','inv_purity','nmi','goodness','jaccard','balacc','homog','compl','vmeasure','falsealarm','fm','q2'}
LESS_IS_BETTER = {'vi','entropy','epratio'}

# Number of times each scenario is repeated (to reduce randomness impact on results)
RUN_NUMBER = 20

# Some file location variables
CLEAN_DATASET_FOLDER = "C:/Users/Vincent/Desktop/Stage PFE/Phase 3 - Tests sur datasets/Datasets propres" # Indicated where all "dataset"_features and "dataset"_classes are located
DATASET_INFORMATION = "C:/Users/Vincent/Desktop/Stage PFE/Phase 3 - Tests sur datasets/dataset_information.txt" # Store dataset information : Dataset - InstanceNumber - DimensionNumber
PREDICTED_CLUSTERING_FOLDER = "C:/Users/Vincent/Desktop/Stage PFE/Phase 3 - Tests sur datasets/Prédiction de clustering/" # Store clustering prediction files : "dataset"_"algorithm"_"runNumber"
RESULTS_FILE = "C:/Users/Vincent/Desktop/Stage PFE/Phase 3 - Tests sur datasets/results.txt" # Store all criterion scores : Dataset - Algorithm - Criteria - RunNumber - Score
TIMES_FILE = "C:/Users/Vincent/Desktop/Stage PFE/Phase 3 - Tests sur datasets/times.txt" # Store execution time for each cases as following : Dataset - Algorithm - RunNumber - InstanceNumber - DimensionNumber - Time - NumberOfIteration(*)
MEAN_RESULTS_FILE = "C:/Users/Vincent/Desktop/Stage PFE/Phase 3 - Tests sur datasets/results_mean.txt" # Store all mean criterion scores : Dataset - Algorithm - Criteria - Score
RANKED_ALGORITHM =  "C:/Users/Vincent/Desktop/Stage PFE/Phase 3 - Tests sur datasets/ranked_algorithm.txt" # Store best algorithm(**) for each case : Criteria - Dataset - Algorithm #1 ( - Algorithm #2 ) ( - ...)

# (*) Some algorithm (currently DBSCAN and Meanshift) need to perform several iteration to predict the correct number of cluster
# (**) Some algorithm may arrive "ex aequo".

def main_part1():
    # 2 actions :
    # - Compute clustering predictions, and save them in the PREDICTED_CLUSTERING_FOLDER under the format "datasetName"_"algorithmName"_"numberOfRun"
    # - Compute all criterion scores for the predictions computed (no mean calculation )

    # Note : main_part2() won't currently work properly if this function is interrupted while computing criterion scores. However, it will if this function is interrupted while processing clustering prediction

    print("MAIN_PART1 : Processing clustering predictions and criterion")

    dataset_list = load_dataset_list(type="compute_prediction_and_metrics")

    count = 1

    for dataset in dataset_list:
        print("DATASET "+str(count)+"/"+str(len(dataset_list))+": "+dataset)

        if not(dataset_already_processed(dataset)): #Check whether number of line with the dataset name in RESULTS_FILE is equal to :
                                # Number of algorithms TIMES Number of criterions TIMES Number of runs per test

            print("\tLOADING DATA")

            #Getting the labels
            classes_file = CLEAN_DATASET_FOLDER + "/" + dataset + "_classes"
            f = open(classes_file, 'r')
            classes_matrix = []
            for line in f:
                classes_matrix = classes_matrix + [[float(num) for num in line.split(',')]]
            labels = np.array(classes_matrix)
            f.close()

            #Getting the features values
            values_file = CLEAN_DATASET_FOLDER + "/" + dataset + "_features"
            f = open(values_file, 'r')
            features_matrix = []
            for line in f:
                features_matrix = features_matrix + [[float(num) for num in line.split(',')]]
            data_values = np.array(features_matrix)
            f.close()

            #Counting number of real clusters
            tmp = np.unique(classes_matrix)
            real_number_cluster = len(tmp)

            #Both process clustering predictions and criterion score calculation
            process_dataset(data_values, real_number_cluster, labels, dataset)
            print("\tDONE")
        else:
            print("\tALREADY PROCESSED")
        count = count+1

def main_part2():
    # Compute for each case the mean value of metrics over the RUN_NUMBER number of prediction for each tuple (algorithm, dataset, criteria)

    dataset_list = load_dataset_list(type="compute_mean_values")

    # Getting all criterion score and already calculated mean criterion score
    with open(MEAN_RESULTS_FILE,"r") as file:
        mean_string = file.read()
    file.close()
    with open(RESULTS_FILE,"r") as file:
        all_results_string = [line.rstrip('\n') for line in file]
    file.close()

    # Compute mean values
    count = 1
    for dataset in dataset_list:
        print("COMPUTE MEAN : "+str(count)+"/"+str(len(dataset_list)))
        for algorithm in ALGORITHM_LIST:
            for criteria in CRITERION_LIST:

                # For each dataset, algorithm and criteria :
                if dataset+"\t"+algorithm+"\t"+criteria not in mean_string: # Mean value not calculated
                    values = [0 for k in range(RUN_NUMBER)]

                    # Getting criterion value for the "nb_run"-th run
                    for nb_run in range(RUN_NUMBER):
                        i=0
                        while i<len(all_results_string) and ((dataset+"\t"+algorithm+"\t"+criteria+"\t"+str(nb_run)) not in all_results_string[i]):
                            i = i+1
                        if i == len(all_results_string):
                            print("ERROR : "+criteria+" score for "+algorithm+" on "+dataset+ " (run "+str(nb_run)+" ) not found in RESULTS_FILE : Did you stop main_part1() earlier while processing criterion score ?")
                        line = (all_results_string[i]).split("\t")
                        values[nb_run] = float(line[4]) # Dataset - Algorithme - Critère - NbTest - =>Valeur<=

                    #Computing mean value
                    mean = np.mean(values)

                    #Saving mean value in MEAN_RESULTS_FILE
                    with open(MEAN_RESULTS_FILE,"a+") as file:
                        file.write(dataset+"\t"+algorithm+"\t"+criteria+"\t"+str(mean)+"\n")
                    file.close()
        count = count+1

def main_part3():
    # Establish algorithm ranking for all cases

    # Note : Despite the warning at the beginning of the file, this function cannot work even if some algorithm or criteria is ADDED in CRITERION_LIST of ALGORITHM. Thus, it is advisable to delete RANKED_ALGORITHM file content each time this function is called

    dataset_list = load_dataset_list(type="compute_algorithm_ranking")

    # A test to check if all criterion are either present on MORE_IS_BETTER or LESS_IS_BETTER list
    test = MORE_IS_BETTER.union(LESS_IS_BETTER)
    if (test != CRITERION_LIST):
        print("ERROR : missing criterion in MORE_IS_BETTER or LESS_IS_BETTER list")
        exit(1)

    # Getting both mean values and already ranked algorithm for all cases already processed
    with open(MEAN_RESULTS_FILE) as file:
        mean_string = [line.rstrip('\n') for line in file]
    file.close()
    with open(RANKED_ALGORITHM) as file:
        already_ranked = file.read()
    file.close()

    count = 0
    for criteria in CRITERION_LIST:
        count_bis = 0
        for dataset in dataset_list:
            print("CRITERIA "+str(count+1)+"/"+str(len(CRITERION_LIST))+ "("+criteria+")" + " : DATASET "+str(count_bis+1)+"/"+str(len(dataset_list))+ "("+dataset+")")

            # For each (criteria, dataset) couple not processed :
            if (criteria + "\t" + dataset) not in already_ranked:

                # Getting criteria score for all algorithm
                dico = {} #Dictionnary
                for algorithm in ALGORITHM_LIST:
                    i = 0
                    while i < len(mean_string) and ((dataset + "\t" + algorithm + "\t" + criteria + "\t") not in mean_string[i]):
                        i = i + 1
                    if i == len(mean_string):
                        print("Error : "+criteria+" score for "+algorithm+" prediction on "+dataset+" dataset not found in MEAN_RESULTS_FILE.")
                    line = (mean_string[i]).split("\t")
                    criteria_score = line[3] # dataset - algorithm - criteria - =>VALUE<=
                    dico[algorithm] = criteria_score

                # Ranking algorithm
                if criteria in MORE_IS_BETTER:
                    ranked_algorithm = sorted(dico.items(), key = lambda x : x[1],reverse=True)
                else:
                    ranked_algorithm = sorted(dico.items(), key=lambda x: x[1])

                # Getting the best (or the best "ex aequo") algorithm(s)
                best_algorithms = ""
                best_value = ranked_algorithm[0][1]
                for couple in ranked_algorithm:
                    if couple[1] == best_value:
                        best_algorithms = couple[0] + "\t" + best_algorithms

                # Saving the best algorithm(s)
                with open(RANKED_ALGORITHM,"a+") as file:
                    file.write(criteria+"\t"+dataset+"\t"+best_algorithms+"\n")
                file.close()

            count_bis = count_bis +1
        count = count+1

def get_objectives_progress():
    # Give graphical representation of RANKED_ALGORITHM file

    # Getting best algorithm for each case
    with open(RANKED_ALGORITHM) as file:
        lines = [line.rstrip('\n') for line in file]
    file.close()

    index = 1

    # Creating seperated graphs for each algorithm
    for algo in ALGORITHM_LIST:
        #Initialize all counts at 0
        dico_algo = dict.fromkeys(CRITERION_LIST,0)


        for line in lines:
            #For each line where the algorithm is one of the best :
            if algo in line:
                line = line.split("\t")
                count = len(line) - 3
                # Adding 1/count to the number of dataset where the algorithm is better (if only 1 criteria is considered)
                dico_algo[line[0]] = dico_algo.get(line[0]) + (1.0/float(count) ) #Note : If several algorithms are ex aequo, the "weight" of the dataset is equally divided by the number of algorithm

        X_values = []
        Y_values = []
        for key, value in dico_algo.items():
            X_values = X_values + [key]
            Y_values = Y_values + [value]

        plt.figure(index)
        plt.bar(X_values,Y_values,align='center')
        plt.hlines(5,0,len(CRITERION_LIST)) # 5 is the target number for each criteria (and for each algorithm)
        plt.title(algo)
        index = index + 1
        plt.savefig("image"+str(index)+".png")
    plt.show()

def generate_information_file():
    # Generate INFORMATION_FILE, which contain list of "clean" datasets, with number of dimension and instance

    # Deleting previous list
    open(DATASET_INFORMATION, 'w').close()

    # Looking at CLEAN_DATASET_FOLDER
    datasets = os.listdir(CLEAN_DATASET_FOLDER)
    for i in range(len(datasets)):
        datasets[i] = (datasets[i]).replace("_features","")
        datasets[i] = (datasets[i]).replace("_classes","")
    datasets = set(datasets)

    # Getting for each found dataset its characteristics
    for dataset in datasets:
        values_file = CLEAN_DATASET_FOLDER + "/" + dataset + "_features"
        f = open(values_file, 'r')
        features_matrix = []
        for line in f:
            features_matrix = features_matrix + [[float(num) for num in line.split(',')]]
        f.close()
        values = np.array(features_matrix)
        size = np.shape(values)

        # Saving information
        with open(DATASET_INFORMATION,"a+") as file:
            file.write(dataset + "\t" + str(size[0]) + "\t" + str(size[1]) + "\n")
        file.close()

        # Fix : Remove last "\n"
        print("Done. Remove the empty line generated in file.")



def load_dataset_list(type):
    # Return the available datasets list. The considered "avaibable" dataset may vary for main_part1, main_part2 and main_part3

    # main_part1() : Simply get all datasets listed in DATASET_INFORMATION (ie available in CLEAN_DATASET_FOLDER)
    if type == "compute_prediction_and_metrics":
        with open(DATASET_INFORMATION,"r") as ds_info:
            string = ds_info.read()
            lines = string.split("\n")
            dictio = {}
            for line in lines:
                values = line.split("\t")
                dictio[str(values[0])] = int(values[1]) * int(values[2])
            result = sorted(dictio.items(), key=lambda x: x[1])  # Ordering datasets by DimensionNumber X InstanceNumber ascending, to process easier datasets first
            print(result)
            dataset_list = []
        for i in result:
            dataset_list = dataset_list + [i[0]]
        ds_info.close()
        return dataset_list

    # main_part2() : Get all datasets available in CLEAN_DATASET_FOLDER, but remove datasets when no mentionned in RESULTS_FILE (ie when not a single criteria score calculation occured about this dataset)
    if type == "compute_mean_values":
        old_dataset_list = load_dataset_list("compute_prediction_and_metrics")
        new_dataset_list = copy.deepcopy(old_dataset_list)

        with open(RESULTS_FILE,"r") as file:
            string_result = file.read()
        file.close()

        for dataset in old_dataset_list:
            if ("\n"+dataset+"\t") not in string_result:
                new_dataset_list.remove(dataset)

        return new_dataset_list

    # main_part3() : Get all datasets available in CLEAN_DATASET_FOLDER, but remove datasets when no mentionned in MEAN_RESULTS_FILE
    if type == "compute_algorithm_ranking":
        old_dataset_list = load_dataset_list("compute_prediction_and_metrics")
        new_dataset_list = copy.deepcopy(old_dataset_list)

        with open(MEAN_RESULTS_FILE, "r") as file:
            mean_string_result = file.read()
        file.close()

        for dataset in old_dataset_list:
            if ("\n" + dataset + "\t") not in mean_string_result:
                new_dataset_list.remove(dataset)

        return new_dataset_list


def dataset_already_processed(name):
    # Check if a dataset has already been processed, ie if all predictions and all criterion calculation are available.
    # Since criterion calculation induce that all predictions are already calculated, this function simply check if
    # the number of occurences of the dataset is NumberOfAlgorithm X NumberOfCriteria X RunNumber

    with open(RESULTS_FILE,'r') as file:
        string = file.read()
        occurences = string.count(name + "\t")
    file.close()
    if occurences == len(ALGORITHM_LIST) * len(CRITERION_LIST) * RUN_NUMBER:
       return True
    return False

def process_dataset(data_values, real_number_cluster, labels, name):
    # Both compute predictions and calculate criterion scores

    #Create predictions in PREDICTED_CLUSTERING_FOLDER, if not all necessary predictions for the following instructions
    # are already computed
    get_predictions(data_values, real_number_cluster,name)

    #Load the RESULT_FILE, to check the progress of metrics calculation
    with open(RESULTS_FILE, "r") as file:
        current_results = file.read()
    file.close()

    for algorithm in ALGORITHM_LIST:
        compute_predictions_metrics(name, algorithm,current_results,labels)

def compute_predictions_metrics(name, algorithm,current_results,labels):
    # Compute for all each (dataset, algorithm) couple the score for all criterion, on the RUN_NUMBER runs
    # Note : This function currently recompute all metrics if at least one value is missing (cf. dataset_already_processed() function)

    with open(RESULTS_FILE,'a+') as result_file:

        # For each of the RUN_NUMBER prediction for this case :
        for run_number in range(RUN_NUMBER):

            # Loading prediction
            print("\tCOMPUTING METRICS VALUES FOR PREDICTIONS WITH "+algorithm+ "("+str(run_number+1)+"/"+str(RUN_NUMBER)+")")
            clustering_prediction = utility.load_csv(PREDICTED_CLUSTERING_FOLDER+name+"_"+algorithm+"_"+str(run_number))

            # Computing metrics for the prediction number "run_number"
            metrics_dictionnary = compute_prediction_metrics(labels,clustering_prediction)

            # Saving values
            for criteria in CRITERION_LIST:
                if current_results.find(name+"\t"+algorithm+"\t"+criteria+"\t"+str(run_number)) == -1:
                    result_file.write(name + "\t" + algorithm + "\t" + criteria + "\t" + str(run_number) + "\t" + str(metrics_dictionnary.get(criteria)) + "\n")
    result_file.close()


def get_predictions(data_values, real_cluster_number,name):
    # Compute for the dataset all required predictions for the experiment : RUN_NUMBER predictions for each algorithm

    for algorithm in ALGORITHM_LIST:
        for run in range(RUN_NUMBER):
            get_prediction(data_values,real_cluster_number,name,run,algorithm)

def get_prediction(data_values, real_cluster_number,name,run_number,algorithm):
    # Compute for the (dataset, algorithm) couple the "run_number" clustering prediction (of RUN_NUMBER)

    # Checking if the "run_number"-th prediction already exists
    prediction_file = PREDICTED_CLUSTERING_FOLDER+name+"_"+algorithm+"_"+str(run_number)
    algorithm_iteration=1
    if os.path.isfile(prediction_file):
        return

    else:
        # Compute prediction

        print("\tGENERATING CLUSTERING PREDICTION WITH "+algorithm+"("+str(run_number+1)+"/"+str(RUN_NUMBER)+")")
        t0 = time.time()

        #SWITCH
        if algorithm == "kmean":
            kmean = sklearn.cluster.KMeans(real_cluster_number).fit(data_values)
            prediction = kmean.labels_
        elif algorithm == "meanshift":
            prediction,algorithm_iteration = get_meanshift_prediction(data_values, real_cluster_number)
        elif algorithm == "complete_agglo":
            agglo_complete = sklearn.cluster.AgglomerativeClustering(real_cluster_number, linkage='complete').fit(data_values)
            prediction = agglo_complete.labels_
        elif algorithm == "ward_agglo":
            agglo_ward = sklearn.cluster.AgglomerativeClustering(real_cluster_number, linkage='ward').fit(data_values)
            prediction = agglo_ward.labels_
        elif algorithm == "dbscan":
            prediction, algorithm_iteration = get_dbscan_prediction(data_values, real_cluster_number)
        else:
            print("ERROR : This code does not support the algorithm : "+algorithm)
        t1 = time.time()

        utility.save_csv(prediction_file,prediction)

        size = np.shape(data_values)

        # Saving results
        with open(TIMES_FILE,'a+') as file:
            file.write(name+"\t"+algorithm+"\t"+str(run_number)+"\t"+str(size[0])+"\t"+str(size[1])+"\t"+str(t1-t0)+"\t"+str(algorithm_iteration)+"\n")
        file.close()


def get_meanshift_prediction(data_values, real_cluster_number, max_bandwidth = 100,count=1):
    # Scikit-Learn module already handle this algorithm, but we needed to encapsulate it into a dichotomic search, to ensure
    # that the number of predicted clusters is correct. It is required since other algorithm (like Kmeans) has access to this information
    # as an input
    # It is possible since there's only 1 numerical input to adjust (bandwidth), and since nb_cluster = f(meanshift(bandwitdh)) is an monotonous (decreasing) function

    print("real_value="+str(real_cluster_number))

    # Checking if the higher bound is correct, otherwise increase it by 10
    low = 0
    high = max_bandwidth
    middle = (low+high)/2 * (random.uniform(0.9,1.1))
    meanshift = sklearn.cluster.MeanShift(bandwidth=middle).fit(data_values)
    prediction = meanshift.labels_
    cluster_number = len(np.unique(prediction))
    print("low="+str(low)+" middle="+str(middle)+" high="+str(high)+" nb="+str(cluster_number))
    if cluster_number > real_cluster_number: #If bandwidth is too small
        return get_meanshift_prediction(data_values,real_cluster_number,max_bandwidth*10,count+1)

    # Processing dichotomic search on the parameter
    while True:
        if cluster_number == real_cluster_number:
            return prediction,count
        elif cluster_number > real_cluster_number: #Bandwidth too small
            low = middle
        elif cluster_number < real_cluster_number: #BW too high
            high = middle
        count = count+1
        middle = (high + low) / 2  + (random.uniform(-0.1,0.1) * (high-low)) # Adding a bit of randomness to ensure each of the RUN_NUMBER runs are differents
        meanshift = sklearn.cluster.MeanShift(bandwidth=middle).fit(data_values)
        prediction = meanshift.labels_
        cluster_number = len(np.unique(prediction))
        print("low=" + str(low) + " middle=" + str(middle) + " high=" + str(high)+" nb="+str(cluster_number))

def get_dbscan_prediction(data_values, real_cluster_number, max_eps = 100,count=1):
    # Scikit-Learn module already handle this algorithm, but we needed to encapsulate it into a dichotomic search, to ensure
    # that the number of predicted clusters is correct. It is required since other algorithm (like Kmeans) has access to this information
    # as an input
    # It is possible since there's only 1 numerical input to adjust (eps) (note : we're not considering outliers), and since nb_cluster = f(dbscan(eps)) is an monotonous (decreasing) function

    # Checking if the higher bound is correct, otherwise increase it by 10
    print("real_value="+str(real_cluster_number))
    low = 0
    high = max_eps
    middle = (low+high)/2 * (random.uniform(0.9,1.1))
    dbscan = sklearn.cluster.DBSCAN(eps=middle,min_samples=1).fit(data_values)
    prediction = dbscan.labels_
    cluster_number = len(np.unique(prediction))
    print("low="+str(low)+" middle="+str(middle)+" high="+str(high)+" nb="+str(cluster_number))
    if cluster_number > real_cluster_number: #If bandwidth is too small
        return get_meanshift_prediction(data_values,real_cluster_number,max_eps*10,count+1)

    # Processing dichotomic search on the parameter
    while True:
        if cluster_number == real_cluster_number:
            return prediction,count
        elif cluster_number > real_cluster_number: #Bandwidth too small
            low = middle
        elif cluster_number < real_cluster_number: #BW too high
            high = middle
        count = count+1
        middle = (high + low) / 2 + (random.uniform(-0.1,0.1) * (high-low))
        dbscan = sklearn.cluster.DBSCAN(eps=middle,min_samples=1).fit(data_values)
        prediction = dbscan.labels_
        cluster_number = len(np.unique(prediction))
        print("low=" + str(low) + " middle=" + str(middle) + " high=" + str(high)+" nb="+str(cluster_number))


def compute_prediction_metrics(y_true, y_pred):
    # Calculate criteria score for each one present in CRITERION_LIST

    # Calculating values that are presents in several formula of criterion
    contingency = utility.compute_contingency(y_true, y_pred)
    a_i = utility.compute_a_i(contingency)
    b_j = utility.compute_b_j(contingency)
    true_clustering_entropy = sklearn.metrics.cluster.entropy(y_true)
    predicted_clustering_entropy = sklearn.metrics.cluster.entropy(y_pred)
    PBV = utility.pair_based_values(contingency)

    # Getting all criterion values
    mi = converted_metrics.mutual_information(y_true, y_pred, contingency, a_i, b_j)
    ari = converted_metrics.adjusted_rand_index(y_true, y_pred, contingency, PBV, a_i, b_j)
    ami = converted_metrics.adjusted_mutual_information(y_true, y_pred, contingency, a_i, b_j, mi,
                                                        true_clustering_entropy, predicted_clustering_entropy)
    compl = converted_metrics.homogeneity(y_true, y_pred, contingency, mi, predicted_clustering_entropy)
    homog = converted_metrics.completness(y_true, y_pred, contingency, mi, true_clustering_entropy)
    vmeasure = converted_metrics.v_measure(y_true, y_pred, 1, mi, contingency, predicted_clustering_entropy,
                                           true_clustering_entropy, homog, compl)
    entropy = new_metrics.cond_entropy(y_true, y_pred, contingency, a_i)
    #accuracy = converted_metrics.accuracy(y_true, y_pred, contingency, PBV)  Is equal to ARI in pair-based value context
    precision = converted_metrics.precision(y_true, y_pred, contingency, PBV)
    recall = converted_metrics.recall(y_true, y_pred, contingency, PBV)
    falsealarm = converted_metrics.false_alarm_rate(y_true, y_pred, contingency, PBV)
    fm = converted_metrics.fowlkes_mallows(y_true, y_pred, contingency, PBV)
    f1 = converted_metrics.f_beta_score(y_true, y_pred, 1, contingency, PBV, precision, recall)
    purity = new_metrics.purity(y_true, y_pred, contingency)
    inversed_purity = new_metrics.inversed_purity(y_true, y_pred, contingency)
    epratio = new_metrics.ep_ratio(y_true, y_pred, contingency, a_i, entropy, purity)
    jaccard = converted_metrics.jaccard_index(y_true, y_pred, contingency, PBV)
    nmi = converted_metrics.normalized_mutual_information(y_true, y_pred, contingency, a_i, b_j, mi,
                                                          true_clustering_entropy, predicted_clustering_entropy)
    ri = new_metrics.rand_index(y_true, y_pred, contingency, PBV)
    vi = new_metrics.variation_of_information(y_true, y_pred, contingency, a_i, b_j)
    # clustering error not calculated : always equal to 1 - accuracy
    goodness = converted_metrics.goodness(y_true, y_pred, contingency, PBV)
    bal_accuracy = converted_metrics.balanced_accuracy(y_true, y_pred, contingency, PBV)
    q2 = new_metrics.q2(y_true, y_pred, contingency, entropy, a_i, b_j)

    metrics_dictionnary = {
        "mi": mi,
        "ari": ari,
        "ami": ami,
        "compl": compl,
        "homog": homog,
        "vmeasure": vmeasure,
        "entropy": entropy,
        "precision": precision,
        "recall": recall,
        "falsealarm": falsealarm,
        "fm": fm,
        "f1": f1,
        "purity": purity,
        "inv_purity": inversed_purity,
        "epratio": epratio,
        "jaccard": jaccard,
        "nmi": nmi,
        "ri": ri,
        "vi": vi,
        "goodness": goodness,
        "balacc": bal_accuracy,
        "q2": q2
    }

    if set(metrics_dictionnary.keys()) != CRITERION_LIST:
        print("ERROR : One or several criterion are not computed in main.compute_prediction_metrics")

    return metrics_dictionnary

main_part1()