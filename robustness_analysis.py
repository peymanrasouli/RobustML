import warnings
warnings.filterwarnings("ignore")
import time
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('display.width', 1000)
import seaborn as sns
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
from prepare_datasets import *
from model_construction import ModelConstruction
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import torch
from torch.autograd import Variable
from progress_bar import printProgressBar
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from LowProFool.Adverse import lowProFool, deepfool
from LowProFool.Metrics import *
import sys
sys.path.insert(0, "CARE")
from CARE.care.care import CARE
from CARE.care_explainer import CAREExplainer


def main():
    # defining path of data sets and experiment results
    path = './'
    dataset_path = path + 'datasets/'
    experiment_path = path + 'experiments/'

    # defining the list of data sets
    datsets_list = {
        'adult': ('adult.csv', PrepareAdult, 'classification'),
        'credit-card_default': ('credit-card-default.csv', PrepareCreditCardDefault, 'classification'),
        'compas': ('compas-scores-two-years.csv', PrepareCOMPAS, 'classification'),
        'german-credit': ('german-credit.csv', PrepareGermanCredit, 'classification'),
        'heart-disease': ('heart-disease.csv', PrepareHeartDisease, 'classification'),
    }

    # defining the list of black-boxes
    blackbox_list = {
        'nn': MLPClassifier,
        # 'gb': GradientBoostingClassifier,
        # 'svc': SVC,
        # 'rf': RandomForestClassifier,
        # 'dt': DecisionTreeClassifier
    }

    for dataset_kw in datsets_list:
        # reading the data set
        dataset_name, prepare_dataset_fn, task = datsets_list[dataset_kw]
        dataset = prepare_dataset_fn(dataset_path, dataset_name)

        # splitting the data set into train and test sets
        X, y = dataset['X_ord'], dataset['y']
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for blackbox_name, blackbox_constructor in blackbox_list.items():
            print('dataset=', dataset_kw)
            print('blackbox=', blackbox_name)
            print('\n')

            # creating the black-box model
            blackbox, \
            train_performance, \
            test_performance = ModelConstruction( X_train, X_test, Y_train, Y_test, blackbox_name, blackbox_constructor)
            if blackbox_name == 'nn':
                predict_fn = lambda x: blackbox.predict((Variable(torch.tensor(x).float()))).ravel()
                predict_proba_fn = lambda x: blackbox.predict_proba((Variable(torch.tensor(x).float())))
            else:
                predict_fn = lambda x: blackbox.predict(x).ravel()
                predict_proba_fn = lambda x: blackbox.predict_proba(x)

            # creating CARE explainer
            care_explainer = CARE(dataset,
                                 task='classification',
                                 predict_fn=predict_fn,
                                 predict_proba_fn=predict_proba_fn,
                                 SOUNDNESS=False,
                                 COHERENCY=False,
                                 ACTIONABILITY=False,
                                 n_cf=1,
                                 n_population=100,
                                 n_generation=20,
                                 x_init=0.5,
                                 neighbor_init=0.5,
                                 random_init=0.5,
                                 K_nbrs=100)
            care_explainer.fit(X_train, Y_train)


            def get_weights(df, target, show_heatmap=False):
                def heatmap(cor):
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
                    plt.show()

                cor = df.corr()
                cor_target = abs(cor[target])

                weights = cor_target[:-1]  # removing target WARNING ASSUMES TARGET IS LAST
                weights = weights / np.linalg.norm(weights)
                if show_heatmap:
                    heatmap(cor)

                return weights.values

            def gen_adv(config, method):
                df_test = config['TestData']
                model = config['Model']
                weights = config['Weights']
                bounds = config['Bounds']
                maxiters = config['MaxIters']
                alpha = config['Alpha']
                lambda_ = config['Lambda']
                feature_names = config['FeatureNames']
                care_explainer = config['CARE_explainer']

                results = {'X_orig': [],
                           'X_adv': [],
                           'Time': []}

                for _, x in df_test.iterrows():

                    x_orig = x[feature_names].to_numpy().reshape(1, -1).ravel()
                    x_tensor = torch.FloatTensor(x[feature_names])

                    if method == 'LowProFool':
                        start_time = time.time()
                        orig_pred, adv_pred, x_adv, loop_i = lowProFool(x_tensor, model, weights, bounds,
                                                                        maxiters, alpha, lambda_)
                        spent_time = time.time() - start_time
                    elif method == 'DeepFool':
                        start_time = time.time()
                        orig_pred, adv_pred, x_adv, loop_i = deepfool(x_tensor, model, maxiters, alpha,
                                                                      bounds, weights=[])
                        x_adv = x_adv.cpu().numpy()
                        spent_time = time.time() - start_time
                    elif method == 'CARE':
                        start_time = time.time()
                        explanation = care_explainer.explain(x_orig)
                        x_adv = explanation['best_cf_ord'].to_numpy()
                        spent_time = time.time() - start_time
                    else:
                        raise Exception("Invalid method", method)

                    results['X_orig'].append(x_orig)
                    results['X_adv'].append(x_adv)
                    results['Time'].append(spent_time)

                return results

            # creating data frames from the train and test data
            X_train_df = pd.DataFrame(data=np.c_[X_train, Y_train], columns=dataset['feature_names'] + ['class'])
            X_test_df = pd.DataFrame(data=np.c_[X_test, Y_test], columns=dataset['feature_names'] + ['class'])

            # extracting the data's lower and upper bounds
            bounds = [np.min(X_train, axis=0), np.max(X_test, axis=0)]

            # computing the weights to model the expert's knowledge
            weights = get_weights(X_train_df, 'class')

            # building experimental config
            config = {'Dataset': dataset_kw,
                      'MaxIters': 10000,
                      'Alpha': 0.001,
                      'Lambda': 0.0,
                      'TrainData': X_train_df,
                      'TestData': X_test_df,
                      'FeatureNames': dataset['feature_names'],
                      'Target': dataset['class_name'],
                      'Weights': weights,
                      'Bounds': bounds,
                      'Model':blackbox,
                      'Predict_fn': predict_fn,
                      'Predict_proba_fn': predict_proba_fn,
                      'CARE_explainer': care_explainer}


            # sub-sampling to evaluate the robustness
            N = int(0.2 * X_test.shape[0])
            config['TestData'] = config['TestData'].sample(n=N, random_state=42)

            # generating adversarial examples
            print('LowProFool is in progress ...')
            results_lpf = gen_adv(config, 'LowProFool')
            print('DeepFool is in progress ...')
            results_df = gen_adv(config, 'DeepFool')
            print('CARE is in progress ...')
            results_care = gen_adv(config, 'CARE')
            config['AdvData'] = {'LowProFool': results_lpf, 'DeepFool': results_df, 'CARE': results_care}

            # evaluating the performance of baseline methods
            def evaluate_performance(config):

                performance = pd.DataFrame(columns=['Method', 'SuccessRate',
                                                    'MeanNormPerturbation', 'StdNormPerturbation',
                                                    'MeanProbability', 'StdProbability',
                                                    'MeanNearestNeighbor', 'StdNearestNeighbor',
                                                    'MeanKNNN', 'StdKNN',
                                                    'MeanTime', 'StdTime'])

                TrainData = config['TrainData']
                X_KNN = TrainData.values[:,:-1]
                Y_KNN = TrainData.values[:, -1].astype(int)

                KNN = [[],[]]
                for c in [0,1]:
                    ind_c = np.where(Y_KNN==c)[0]
                    X_c = X_KNN[ind_c,:]
                    Y_c = Y_KNN[ind_c]
                    KNN[c] = NearestNeighbors(n_neighbors=50).fit(X_c, Y_c)

                for method, results in config['AdvData'].items():

                    metrics = []
                    X_orig = np.asarray(results['X_orig'])
                    X_adv = np.asarray(results['X_adv'])
                    predict_fn = config['Predict_fn']
                    predict_proba_fn =  config['Predict_proba_fn']

                    # method's name
                    metrics.append(method)

                    # success rate
                    pred_orig = predict_fn(X_orig)
                    pred_adv =predict_fn(X_adv)
                    success_rate = np.round(np.mean(pred_orig != pred_adv),3)
                    metrics.append(success_rate)

                    # norm perturbations
                    perturbations = np.abs(X_orig - X_adv)
                    norm_perturbations =  np.linalg.norm(perturbations, axis=1)
                    mean_norm_perturbations = np.round(np.mean(norm_perturbations),3)
                    std_norm_perturbations = np.round(np.std(norm_perturbations),3)
                    metrics.append(mean_norm_perturbations)
                    metrics.append(std_norm_perturbations)

                    # probability
                    target_class = 1 - pred_orig
                    proba_adv = predict_proba_fn(X_adv)
                    proba_adv = [proba_adv[i,t] for i,t in enumerate(target_class)]
                    mean_proba = np.round(np.mean(proba_adv),3)
                    std_proba = np.round(np.std(proba_adv),3)
                    metrics.append(mean_proba)
                    metrics.append(std_proba)

                    # distance to nearest neighbor and K-nearest neighbor
                    dist_to_nn = []
                    dist_to_knn = []
                    for x, c in zip(X_adv, target_class):
                        distances, indices = KNN[c].kneighbors(x.reshape(1,-1))
                        dist_to_nn.append(distances[0][0])
                        dist_to_knn.append(np.mean(distances[0]))

                    mean_nearest_neighbor = np.round(np.mean(dist_to_nn),3)
                    std_nearest_neighbor = np.round(np.std(dist_to_nn),3)
                    metrics.append(mean_nearest_neighbor)
                    metrics.append(std_nearest_neighbor)

                    mean_knn = np.round(np.mean(dist_to_knn),3)
                    std_knn = np.round(np.std(dist_to_knn),3)
                    metrics.append(mean_knn)
                    metrics.append(std_knn)

                    # time complexity
                    exe_time = np.asarray(results['Time'])
                    mean_exe_time = np.round(np.mean(exe_time),3)
                    std_exe_time = np.round(np.std(exe_time),3)
                    metrics.append(mean_exe_time)
                    metrics.append(std_exe_time)

                    performance = performance.append(pd.DataFrame(columns=performance.columns,
                                                                  data=np.asarray(metrics).reshape(1,-1)),
                                                     ignore_index=True)
                return performance

            print('\n')
            performance = evaluate_performance(config)
            print(performance)
            print('\n')

if __name__ == '__main__':
    main()