import warnings
warnings.filterwarnings("ignore")
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
            test_performance = ModelConstruction( X_train, X_test, Y_train, Y_test, blackbox_name, blackbox_constructor)
            if blackbox_name == 'nn':
                predict_fn = lambda x: blackbox.predict((Variable(torch.tensor(x).float()))).ravel()
                predict_proba_fn = lambda x: blackbox.predict_proba((Variable(torch.tensor(x).float())))
            else:
                predict_fn = lambda x: blackbox.predict(x).ravel()
                predict_proba_fn = lambda x: blackbox.predict_proba(x)


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
                df_eval = config['EvalData']
                extra_cols = ['orig_pred', 'adv_pred', 'iters']
                model = config['Model']
                weights = config['Weights']
                bounds = config['Bounds']
                maxiters = config['MaxIters']
                alpha = config['Alpha']
                lambda_ = config['Lambda']
                feature_names = config['FeatureNames']

                results = np.zeros((len(df_eval), len(feature_names) + len(extra_cols)))
                i = -1
                for _, x in df_eval.iterrows():
                    i += 1
                    x_tensor = torch.FloatTensor(x[feature_names])

                    if method == 'LowProFool':
                        orig_pred, adv_pred, x_adv, loop_i = lowProFool(x_tensor, model, weights, bounds,
                                                                        maxiters, alpha, lambda_)
                    elif method == 'Deepfool':
                        orig_pred, adv_pred, x_adv, loop_i = deepfool(x_tensor, model, maxiters, alpha,
                                                                      bounds, weights=[])
                    else:
                        raise Exception("Invalid method", method)
                    results[i] = np.concatenate((x_adv, [orig_pred, adv_pred, loop_i]), axis=0)

                return pd.DataFrame(results, index=df_eval.index, columns=feature_names + extra_cols)

            # scale the data and extract the lower and upper bounds
            scaler = MinMaxScaler()
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            bounds = [np.min(X_train_scaled, axis=0), np.max(X_train_scaled, axis=0)]
            X_train_scaled_df = pd.DataFrame(data=np.c_[X_train_scaled, Y_train],
                                             columns=dataset['feature_names'] + ['class'])
            X_test_scaled_df = pd.DataFrame(data=np.c_[X_test_scaled, Y_test],
                                            columns=dataset['feature_names'] + ['class'])

            # Compute the weights to model the expert's knowledge
            weights = get_weights(X_train_scaled_df, 'class')

            # Build experimental config
            config = {'Dataset': dataset_kw,
                      'MaxIters': 20000,
                      'Alpha': 0.001,
                      'Lambda': 4.0,
                      'TrainData': X_train_scaled_df,
                      'TestData': X_test_scaled_df,
                      'Scaler': scaler,
                      'FeatureNames': dataset['feature_names'],
                      'Target': dataset['class_name'],
                      'Weights': weights,
                      'Bounds': bounds,
                      'Model':blackbox}


            # Sub sample to evaluate the robustness
            N = 10
            config['EvalData'] = config['TestData'].sample(n=N, random_state=42)

            # Generate adversarial examples
            df_adv_lpf = gen_adv(config, 'LowProFool')
            df_adv_df = gen_adv(config, 'Deepfool')
            config['AdvData'] = {'LowProFool': df_adv_lpf, 'Deepfool': df_adv_df}

            # Compute metrics
            list_metrics = {'SuccessRate': True,
                            'iter_means': False,
                            'iter_std': False,
                            'normdelta_median': False,
                            'normdelta_mean': True,
                            'n_std': True,
                            'weighted_median': False,
                            'weighted_mean': True,
                            'w_std': True,
                            'mean_dists_at_org': False,
                            'median_dists_at_org': False,
                            'mean_dists_at_tgt': False,
                            'mean_dists_at_org_weighted': True,
                            'mdow_std': True,
                            'median_dists_at_org_weighted': False,
                            'mean_dists_at_tgt_weighted': True,
                            'mdtw_std': True,
                            'prop_same_class_arg_org': False,
                            'prop_same_class_arg_adv': False}

            all_metrics = get_metrics(config, list_metrics)
            all_metrics = pd.DataFrame(all_metrics, columns=['Method'] + [k for k, v in list_metrics.items() if v])
            print(all_metrics)



if __name__ == '__main__':
    main()