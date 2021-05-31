import warnings
warnings.filterwarnings("ignore")
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('display.width', 1000)
import shap
import seaborn as sns
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
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
from progress_bar import printProgressBar
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
        # 'credit-card_default': ('credit-card-default.csv', PrepareCreditCardDefault, 'classification'),
        # 'compas': ('compas-scores-two-years.csv', PrepareCOMPAS, 'classification'),
        # 'german-credit': ('german-credit.csv', PrepareGermanCredit, 'classification'),
        # 'heart-disease': ('heart-disease.csv', PrepareHeartDisease, 'classification'),
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
            test_performance = ModelConstruction(dataset, X_train, X_test, Y_train, Y_test,
                                                 blackbox_name, blackbox_constructor)
            predict_fn = lambda x: blackbox.predict(x).ravel()
            predict_proba_fn = lambda x: blackbox.predict_proba(x)



if __name__ == '__main__':
    main()