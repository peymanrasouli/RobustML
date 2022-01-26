import warnings
warnings.filterwarnings("ignore")
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('display.width', 1000)
from prepare_datasets import *
from model_construction import ModelConstruction
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from torch.autograd import Variable
from multiobjective_counterfactual_explanation import MOCE
from generate_perturbations import *
from evaluate_performance import evaluatePerformance

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
    }

    # defining the list of black-boxes
    blackbox_list = {
        'nn': MLPClassifier,
        'gb': GradientBoostingClassifier,
        'svc': SVC,
        # 'rf': RandomForestClassifier,
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
            print('Accuracy of the black-box model:')
            blackbox, \
            train_performance, \
            test_performance = ModelConstruction( X_train, X_test, Y_train, Y_test, blackbox_name, blackbox_constructor)
            if blackbox_name == 'nn':
                predict_fn = lambda x: blackbox.predict((Variable(torch.tensor(x).float()))).ravel()
                predict_proba_fn = lambda x: blackbox.predict_proba((Variable(torch.tensor(x).float())))
            else:
                predict_fn = lambda x: blackbox.predict(x).ravel()
                predict_proba_fn = lambda x: blackbox.predict_proba(x)

            print('Analyzing the robustness of the black-box model:')
            # creating multiobjective counterfactual explainer: MOCE
            MOCE_nonboundary = MOCE(dataset,
                                     predict_fn=predict_fn,
                                     predict_proba_fn=predict_proba_fn,
                                     boundary=False,
                                     n_cf=5,
                                     K_nbrs=100,
                                     n_population=100,
                                     n_generation=50,
                                     crossover_perc=0.6,
                                     mutation_perc=0.4,
                                     hof_size=100,
                                     init_x_perc=0.2,
                                     init_neighbor_perc=1.0,
                                     init_random_perc=0.4)
            MOCE_nonboundary.fit(X_train, Y_train)

            # creating data frames from the train and test data
            X_train_df = pd.DataFrame(data=np.c_[X_train, Y_train], columns=dataset['feature_names'] + ['class'])
            X_test_df = pd.DataFrame(data=np.c_[X_test, Y_test], columns=dataset['feature_names'] + ['class'])

            # extracting the data's lower and upper bounds
            bounds = [np.min(X_train, axis=0), np.max(X_test, axis=0)]

            # computing the weights to model the expert's knowledge
            weights = calculateWeights(X_train_df, 'class')

            # building experimental config
            config = {'Dataset': dataset_kw,
                      'MaxIters': 10000,
                      'Alpha': 0.001,
                      'Lambda': 1.0,
                      'Epsilon': 0.05,
                      'TrainData': X_train_df,
                      'TestData': X_test_df,
                      'FeatureNames': dataset['feature_names'],
                      'Target': dataset['class_name'],
                      'Weights': weights,
                      'Bounds': bounds,
                      'Model':blackbox,
                      'Predict_fn': predict_fn,
                      'Predict_proba_fn': predict_proba_fn,
                      'MOCE': MOCE_nonboundary}

            # sub-sampling to evaluate the robustness
            N = int(0.1 * X_test.shape[0])
            config['TestData'] = config['TestData'].sample(n=N, random_state=42)

            # generating adversarial examples
            print('MOCE is in progress ...')
            results_moce = generatePerturbations(config, 'MOCE')
            config['AdvData'] = {'MOCE': results_moce}

            print('\n')
            performance = evaluatePerformance(config)
            for method, results in performance.items():
                print(method)
                print(results)
                print('\n')

if __name__ == '__main__':
    main()