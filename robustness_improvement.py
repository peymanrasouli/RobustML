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
from sklearn.metrics import pairwise_distances
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from LowProFool.Adverse import lowProFool, deepfool
from LowProFool.Metrics import *
from util_funcs import *
import sys
sys.path.insert(0, "CARE_analysis")
from CARE_analysis.care.care import CARE as CARE_analysis
sys.path.insert(0, "CARE_improvement")
from CARE_improvement.care.care import CARE as CARE_imrpovement


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

            # creating CARE explainer for analysis purpose
            care_analysis = CARE_analysis(dataset,
                                             task='classification',
                                             predict_fn=predict_fn,
                                             predict_proba_fn=predict_proba_fn,
                                             SOUNDNESS=False,
                                             COHERENCY=False,
                                             ACTIONABILITY=False,
                                             n_cf=1,
                                             n_population=100,
                                             n_generation=20,
                                             x_init=0.3,
                                             neighbor_init=0.8,
                                             random_init=0.5,
                                             K_nbrs=100)
            care_analysis.fit(X_train, Y_train)

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
                      'CARE_explainer': care_analysis}


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

            print('\n')
            performance = evaluate_performance(config)
            print(performance)
            print('\n')

            ########################################## Robustness Improvement ########################################
            # making prediction for training data
            Y_hat_train = predict_fn(X_train)
            ind_correct = np.where(Y_train == Y_hat_train)[0]
            X_correct = X_train[ind_correct]
            Y_correct = Y_train[ind_correct]
            P_correct = np.max(predict_proba_fn(X_correct), axis=1)
            X_class = [[], []]
            for c in [0, 1]:
                X_class[c] = X_correct[np.where(Y_correct == c)]

            KNN_groundtruth = [[], []]
            for c in [0, 1]:
                KNN_groundtruth[c] = NearestNeighbors(n_neighbors=1, metric='minkowski', p=2).fit(X_class[c])

            # creating CARE explainer for improvement purpose
            care_improvement = CARE_imrpovement(dataset,
                                                 task='classification',
                                                 predict_fn=predict_fn,
                                                 predict_proba_fn=predict_proba_fn,
                                                 SOUNDNESS=False,
                                                 COHERENCY=False,
                                                 ACTIONABILITY=False,
                                                 n_cf=1,
                                                 n_population=100,
                                                 n_generation=20,
                                                 x_init=0.3,
                                                 neighbor_init=0.8,
                                                 random_init=0.5,
                                                 K_nbrs=100)
            care_improvement.fit(X_train, Y_train)

            print('Generating boundary counterfactuals to improve the inter-class margin:')
            prob_thresh = 0.65
            X_cfs = []
            Y_cfs = []
            D_cfs = []
            for i, x, y, p in zip(range(X_correct.shape[0]), X_correct, Y_correct, P_correct):
                if p <= prob_thresh:

                    explanations = care_improvement.explain(x)
                    cf = explanations['best_cf_ord'].to_numpy()
                    X_cfs.append(cf)
                    Y_cfs.append(y)

                    d_cf_x = pairwise_distances(x.reshape(1,-1), cf.reshape(1,-1), metric='minkowski', p=2)[0][0] + 1.0
                    dist, ind = KNN_groundtruth[1-y].kneighbors(cf.reshape(1,-1))
                    d_cf_class = dist[0][0] + 1.0
                    d_ratio =  d_cf_x /  d_cf_class
                    D_cfs.append(d_ratio)

                printProgressBar(i + 1, X_train.shape[0], prefix='Progress:', suffix='Complete', length=50)

            # retraining the blackbox using improved data (original train data + generated counterfactuals)
            n_bins = 8
            bins = np.linspace(min(D_cfs), max(D_cfs), n_bins)
            X_cfs = np.asarray(X_cfs)
            Y_cfs = np.asarray(Y_cfs)

            for b in range(1, n_bins):
                print('\n')
                print('Robustness of improved blackbox using counterfactuals within range bin --%d--:' % (b))
                print('\n')

                selected_cfs = np.where(D_cfs <= bins[b])[0]
                X_add = X_cfs[selected_cfs].copy()
                Y_add = Y_cfs[selected_cfs].copy()

                X_add = np.vstack(X_add)
                Y_add = np.hstack(Y_add)

                X_improved = np.r_[X_train, X_add]
                Y_improved = np.r_[Y_train, Y_add]

                improved_blackbox, \
                improved_train_performance, \
                improved_test_performance = ModelConstruction(X_improved, X_test, Y_improved, Y_test, blackbox_name,
                                                              blackbox_constructor)
                if blackbox_name == 'nn':
                    improved_predict_fn = lambda x: blackbox.predict((Variable(torch.tensor(x).float()))).ravel()
                    improved_predict_proba_fn = lambda x: blackbox.predict_proba((Variable(torch.tensor(x).float())))
                else:
                    improved_predict_fn = lambda x: blackbox.predict(x).ravel()
                    improved_predict_proba_fn = lambda x: blackbox.predict_proba(x)

                # creating CARE explainer for analysis purpose
                care_analysis = CARE_analysis(dataset,
                                              task='classification',
                                              predict_fn=improved_predict_fn,
                                              predict_proba_fn=improved_predict_proba_fn,
                                              SOUNDNESS=False,
                                              COHERENCY=False,
                                              ACTIONABILITY=False,
                                              n_cf=1,
                                              n_population=100,
                                              n_generation=20,
                                              x_init=0.3,
                                              neighbor_init=0.8,
                                              random_init=0.5,
                                              K_nbrs=100)
                care_analysis.fit(X_train, Y_train)

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
                          'Model': improved_blackbox,
                          'Predict_fn': improved_predict_fn,
                          'Predict_proba_fn': improved_predict_proba_fn,
                          'CARE_explainer': care_analysis}

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

                print('\n')
                performance = evaluate_performance(config)
                print(performance)
                print('\n')


if __name__ == '__main__':
    main()