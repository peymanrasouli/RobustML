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
from util_funcs import *
import sys
sys.path.insert(0, "CARE_analysis")
from CARE_analysis.care.care import CARE as CARE_analysis

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
        'gb': GradientBoostingClassifier,
        'svc': SVC,
        'rf': RandomForestClassifier,
        'dt': DecisionTreeClassifier
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
            # print('LowProFool is in progress ...')
            # results_lpf = gen_adv(config, 'LowProFool')
            # print('DeepFool is in progress ...')
            # results_df = gen_adv(config, 'DeepFool')
            print('CARE is in progress ...')
            results_care = gen_adv(config, 'CARE')
            # config['AdvData'] = {'LowProFool': results_lpf, 'DeepFool': results_df, 'CARE': results_care}
            config['AdvData'] = {'CARE': results_care}

            print('\n')
            performance = evaluate_performance(config)
            for method, results in performance.items():
                print(method)
                print(results)
                print('\n')

if __name__ == '__main__':
    main()