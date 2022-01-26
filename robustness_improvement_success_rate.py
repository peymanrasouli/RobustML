import warnings
warnings.filterwarnings("ignore")
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('display.width', 1000)
from prepare_datasets import *
from model_construction import ModelConstruction
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from torch.autograd import Variable
from multiobjective_counterfactual_explanation import MOCE
from generate_perturbations import *
from evaluate_performance import evaluatePerformance
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

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
    }

    vulnerable_classes = {
        'adult':{
            'nn': 0,
            # 'gb': 1,
            # 'svc': 0,
            # 'rf': 1,
        },
        'credit-card_default': {
            'nn': 1,
            # 'gb': 1,
            # 'svc': 1,
            # 'rf': 1,
        },
        'compas': {
            'nn': 0,
            # 'gb': 0,
            # 'svc': 0,
            # 'rf': 0,
        },
        'german-credit': {
            'nn': 1,
            # 'gb': 1,
            # 'svc': 1,
            # 'rf': 1,
        }
    }

    range_perturbations = {
        'adult':[0.001, 0.1],
        'credit-card_default': [0.04, 0.12],
        'compas': [0.001, 0.1],
        'german-credit': [0.2, 0.4]
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
            print('Accuracy of the original NN model:')
            blackbox, \
            train_performance, \
            test_performance = ModelConstruction( X_train, X_test, Y_train, Y_test, blackbox_name, blackbox_constructor)
            if blackbox_name == 'nn':
                predict_fn = lambda x: blackbox.predict((Variable(torch.tensor(x).float()))).ravel()
                predict_proba_fn = lambda x: blackbox.predict_proba((Variable(torch.tensor(x).float())))
            else:
                predict_fn = lambda x: blackbox.predict(x).ravel()
                predict_proba_fn = lambda x: blackbox.predict_proba(x)

            print('Analyzing the success rate of baselines on the original NN model:')
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
                      'MOCE': None}

            # sub-sampling to evaluate the robustness
            N = int(0.1 * X_test.shape[0])
            config['TestData'] = config['TestData'].sample(n=N, random_state=42)

            # generating adversarial examples
            print('LowProFool is in progress ...')
            results_lpf = generatePerturbations(config, 'LowProFool')
            print('DeepFool is in progress ...')
            results_df = generatePerturbations(config, 'DeepFool')
            config['AdvData'] = {'LowProFool': results_lpf, 'DeepFool': results_df}
            print('\n')

            # measuring the success rate w.r.t. different values of epsilon
            vul_class = vulnerable_classes[dataset_kw][blackbox_name]
            min_perturbations = range_perturbations[dataset_kw][0]
            max_perturbations = range_perturbations[dataset_kw][1]
            epsilon_success_rate_original = {'LowProFool': [], 'DeepFool': []}
            for epsilon in np.linspace(min_perturbations,max_perturbations, 40):
                config['Epsilon'] = epsilon
                performance = evaluatePerformance(config)
                for method, results in performance.items():
                    epsilon_success_rate_original[method].append(results.iloc[int(1+vul_class),1])

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
                KNN_groundtruth[c] = NearestNeighbors(n_neighbors=5, metric='minkowski', p=2).fit(X_class[c])

            # creating multiobjective counterfactual explainer: MOCE
            MOCE_boundary = MOCE(dataset,
                                 predict_fn=predict_fn,
                                 predict_proba_fn=predict_proba_fn,
                                 boundary=True,
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
            MOCE_boundary.fit(X_train, Y_train)

            print('Generating boundary counterfactuals to improve the inter-class margin of the NN model:')
            pb = ProgressBar(total=X_correct.shape[0], prefix='Progress:', suffix='Complete',
                             decimals=1, length=50, fill='█', zfill='-')
            vul_class = vulnerable_classes[dataset_kw][blackbox_name]
            prob_thresh = 0.65
            X_cfs = []
            Y_cfs = []
            D_cfs = []
            for i, x, y, p in zip(range(X_correct.shape[0]), X_correct, Y_correct, P_correct):

                if p <= prob_thresh and y == vul_class:

                    explanations = MOCE_boundary.explain(x)
                    cf = explanations['best_cf_ord'].to_numpy()
                    X_cfs.append(cf)
                    Y_cfs.append(y)

                    d_cf_x = pairwise_distances(x.reshape(1,-1), cf.reshape(1,-1), metric='minkowski', p=2)[0][0]
                    dist, ind = KNN_groundtruth[1-y].kneighbors(cf.reshape(1,-1))
                    d_cf_class= np.mean(dist)
                    d_ratio = d_cf_x / d_cf_class
                    D_cfs.append(d_ratio)

                pb.print_progress_bar(i + 1)

            # retraining the blackbox using improved data (original train data + generated counterfactuals)
            n_bins = 8
            bins = np.linspace(min(D_cfs), 1, n_bins)
            X_cfs = np.asarray(X_cfs)
            Y_cfs = np.asarray(Y_cfs)

            for b in range(1, n_bins):
                print('\n')
                print('Analyzing the improved NN model retrained via counterfactuals within '
                      'range bin --%d-- having distance ratio <= --%.3f--:' % (b,bins[b]))
                print('\n')

                selected_cfs = np.where(D_cfs <= bins[b])[0]
                X_add = X_cfs[selected_cfs].copy()
                Y_add = Y_cfs[selected_cfs].copy()

                X_add = np.vstack(X_add)
                Y_add = np.hstack(Y_add)

                X_improved = np.r_[X_train, X_add]
                Y_improved = np.r_[Y_train, Y_add]

                print('Accuracy of the improved NN model:')
                improved_blackbox, \
                improved_train_performance, \
                improved_test_performance = ModelConstruction(X_improved, X_test, Y_improved, Y_test, blackbox_name,
                                                              blackbox_constructor)
                if blackbox_name == 'nn':
                    improved_predict_fn = lambda x: improved_blackbox.predict((Variable(torch.tensor(x).float()))).ravel()
                    improved_predict_proba_fn = lambda x: improved_blackbox.predict_proba((Variable(torch.tensor(x).float())))
                else:
                    improved_predict_fn = lambda x: improved_blackbox.predict(x).ravel()
                    improved_predict_proba_fn = lambda x: improved_blackbox.predict_proba(x)

                print('Analyzing the success rate of baselines on the improved NN model:')
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
                          'Model': improved_blackbox,
                          'Predict_fn': improved_predict_fn,
                          'Predict_proba_fn': improved_predict_proba_fn,
                          'MOCE': None}

                # sub-sampling to evaluate the robustness
                N = int(0.1 * X_test.shape[0])
                config['TestData'] = config['TestData'].sample(n=N, random_state=42)

                # generating adversarial examples
                print('LowProFool is in progress ...')
                results_lpf = generatePerturbations(config, 'LowProFool')
                print('DeepFool is in progress ...')
                results_df = generatePerturbations(config, 'DeepFool')
                config['AdvData'] = {'LowProFool': results_lpf, 'DeepFool': results_df}

                # measuring the success rate w.r.t. different values of epsilon
                epsilon_success_rate_improved = {'LowProFool': [], 'DeepFool': []}
                for epsilon in np.linspace(min_perturbations, max_perturbations, 40):
                    config['Epsilon'] = epsilon
                    performance = evaluatePerformance(config)
                    for method, results in performance.items():
                        epsilon_success_rate_improved[method].append(results.iloc[int(1+vul_class), 1])

                # plot the epsilon success rate
                plt.figure(figsize=(7, 4))
                plt.plot(np.linspace(min_perturbations, max_perturbations, 40),
                         epsilon_success_rate_original['LowProFool'], linestyle='dashed', linewidth=1, color='#FF0000')
                plt.plot(np.linspace(min_perturbations, max_perturbations, 40),
                         epsilon_success_rate_original['DeepFool'],  linestyle='dashed', linewidth=1, color='#0000FF')
                plt.plot(np.linspace(min_perturbations, max_perturbations, 40),
                         epsilon_success_rate_improved['LowProFool'], linewidth=1, color='#FF0000')
                plt.plot(np.linspace(min_perturbations, max_perturbations, 40),
                         epsilon_success_rate_improved['DeepFool'], linewidth=1, color='#0000FF')
                plt.xlabel('epsilon ($\epsilon$)')
                plt.ylabel('success rate')
                plt.legend(['LowProFool-NN$_{original}$', 'DeepFool-NN$_{original}$','LowProFool-NN$_{improved}$', 'DeepFool-NN$_{improved}$'])
                plt.grid()
                plt.savefig(experiment_path + 'success_rate_' + dataset_kw +
                            '_' + blackbox_name + '_' + 'bin_' + str(b) + '.pdf',  bbox_inches='tight')
                plt.show(block=False)
                plt.close()

if __name__ == '__main__':
    main()