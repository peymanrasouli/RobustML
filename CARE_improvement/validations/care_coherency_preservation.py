import os
import sys
sys.path.append("../")
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('display.width', 1000)
from prepare_datasets import *
from utils import *
from sklearn.model_selection import train_test_split
from create_model import CreateModel, MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from care.care import CARE
from user_preferences import userPreferences
from evaluate_counterfactuals import evaluateCounterfactuals

def main():
    # defining path of data sets and experiment results
    path = '../'
    dataset_path = path + 'datasets/'
    experiment_path = path + 'experiments/'

    # defining the list of data sets
    datsets_list = {
        'adult': ('adult.csv', PrepareAdult, 'classification'),
    }

    # defining the list of black-boxes
    blackbox_list = {
        'nn-c': MLPClassifier,
        'gb-c': GradientBoostingClassifier
    }

    experiment_size = {
        'adult': (500, 10),
    }

    for dataset_kw in datsets_list:
        print('dataset=', dataset_kw)
        print('\n')

        # reading a data set
        dataset_name, prepare_dataset_fn, task = datsets_list[dataset_kw]
        dataset = prepare_dataset_fn(dataset_path,dataset_name)

        # splitting the data set into train and test sets
        X, y = dataset['X_ord'], dataset['y']
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for blackbox_name, blackbox_constructor in blackbox_list.items():
            print('blackbox=', blackbox_name)

            # creating black-box model
            blackbox = CreateModel(dataset, X_train, X_test, Y_train, Y_test, task, blackbox_name, blackbox_constructor)
            if blackbox_name == 'nn-c':
                predict_fn = lambda x: blackbox.predict_classes(x).ravel()
                predict_proba_fn = lambda x: np.asarray([1 - blackbox.predict(x).ravel(), blackbox.predict(x).ravel()]).transpose()
            else:
                predict_fn = lambda x: blackbox.predict(x).ravel()
                predict_proba_fn = lambda x: blackbox.predict_proba(x)

            # setting experiment size for the data set
            N, n_cf = experiment_size[dataset_kw]

            # creating/opening a csv file for storing results
            exists = os.path.isfile(
                experiment_path + 'care_coherency_preservation_%s_%s_eval_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_cf))
            if exists:
                os.remove(experiment_path + 'care_coherency_preservation_%s_%s_eval_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_cf))
            eval_results_csv = open(
                experiment_path + 'care_coherency_preservation_%s_%s_eval_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_cf), 'a')

            header = ['Education','','', '',
                      'Relationship', '', '', '']
            header = ','.join(header)
            eval_results_csv.write('%s\n' % (header))
            header = ['Validity',
                      'Validity+Soundness',
                      'Validity+Soundness+Coherency',
                      'Validity+Soundness+Coherency+Actionability',
                      'Validity',
                      'Validity+Soundness',
                      'Validity+Soundness+Coherency',
                      'Validity+Soundness+Coherency+Actionability']
            header = ','.join(header)
            eval_results_csv.write('%s\n' % (header))
            average = '%s,%s,%s,%s,%s,%s,%s,%s\n' % \
                     ('=average(A4:A1000)', '=average(B4:B1000)', '=average(C4:C1000)', '=average(D4:D1000)',
                      '=average(E4:E1000)', '=average(F4:F1000)', '=average(G4:G1000)', '=average(H4:H1000)')
            eval_results_csv.write(average)
            eval_results_csv.flush()

            # CARE with {validity} config
            care_config_1 = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                 SOUNDNESS=False, COHERENCY=False, ACTIONABILITY=False, n_cf=n_cf)
            care_config_1.fit(X_train, Y_train)

            # CARE with {validity, soundness} config
            care_config_12 = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                  SOUNDNESS=True, COHERENCY=False, ACTIONABILITY=False, n_cf=n_cf)
            care_config_12.fit(X_train, Y_train)

            # CARE with {validity, soundness, coherency} config
            care_config_123 = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                    SOUNDNESS=True, COHERENCY=True, ACTIONABILITY=False, n_cf=n_cf)
            care_config_123.fit(X_train, Y_train)

            # CARE with {validity, soundness, coherency, actionability} config
            care_config_1234 = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                                    SOUNDNESS=True, COHERENCY=True, ACTIONABILITY=True, n_cf=n_cf)
            care_config_1234.fit(X_train, Y_train)

            # explaining instances from test set
            # correlation between education-num and education features
            correlations = [(0,13),
                            (1,3),
                            (2,4),
                            (3,5),
                            (4,6),
                            (5,0),
                            (6,1),
                            (7,2),
                            (8,11),
                            (9,15),
                            (10,8),
                            (11,7),
                            (12,9),
                            (13,12),
                            (14,14),
                            (15,10)]
            explained = 0
            for x_ord in X_test:

                explanation_config_1 = care_config_1.explain(x_ord)
                explanation_config_12 = care_config_12.explain(x_ord)
                explanation_config_123 = care_config_123.explain(x_ord)
                user_preferences = userPreferences(dataset, x_ord)
                explanation_config_1234 = care_config_1234.explain(x_ord, user_preferences=user_preferences)

                # evaluating counterfactuals based on all objectives results
                toolbox = explanation_config_1234['toolbox']
                objective_names = explanation_config_1234['objective_names']
                featureScaler = explanation_config_1234['featureScaler']
                feature_names = dataset['feature_names']

                # evaluating and recovering counterfactuals of {validity} config
                cfs_ord_config_1, \
                cfs_eval_config_1, \
                x_cfs_ord_config_1, \
                x_cfs_eval_config_1 = evaluateCounterfactuals(x_ord, explanation_config_1['cfs_ord'], dataset,
                                                              predict_fn, predict_proba_fn, task, toolbox,
                                                              objective_names, featureScaler, feature_names)

                # Education correlation set
                education_num = cfs_ord_config_1['education-num'].to_numpy().astype(int)
                education = cfs_ord_config_1['education'].to_numpy().astype(int)
                education_preserved_config_1 = 0
                for n in range(n_cf):
                    education_preserved_config_1 += 1 if correlations[education_num[n]][1] == education[n] else 0
                education_preserved_config_1 = education_preserved_config_1 / n_cf

                # Relationship correlation set
                relationship = cfs_ord_config_1['relationship'].to_numpy().astype(int)
                marital_status = cfs_ord_config_1['marital-status'].to_numpy().astype(int)
                sex = cfs_ord_config_1['sex'].to_numpy().astype(int)
                relationship_preserved_config_1 = 0
                for n in range(n_cf):
                    if relationship[n] == 0:
                        relationship_preserved_config_1 += 1 if sex[n]==1 and marital_status[n]==2  else 0
                    elif relationship[n] == 5:
                        relationship_preserved_config_1 += 1 if sex[n]==0 and marital_status[n]==2  else 0
                    else:
                        relationship_preserved_config_1 += 1
                relationship_preserved_config_1 = relationship_preserved_config_1 / n_cf

                # evaluating and recovering counterfactuals of {validity, soundness} config
                cfs_ord_config_12, \
                cfs_eval_config_12, \
                x_cfs_ord_config_12, \
                x_cfs_eval_config_12 = evaluateCounterfactuals(x_ord, explanation_config_12['cfs_ord'], dataset,
                                                               predict_fn, predict_proba_fn, task, toolbox,
                                                               objective_names, featureScaler, feature_names)

                # Education correlation set
                education_num = cfs_ord_config_12['education-num'].to_numpy().astype(int)
                education = cfs_ord_config_12['education'].to_numpy().astype(int)
                education_preserved_config_12 = 0
                for n in range(n_cf):
                    education_preserved_config_12 += 1 if correlations[education_num[n]][1] == education[n] else 0
                education_preserved_config_12 = education_preserved_config_12 / n_cf

                # Relationship correlation set
                relationship = cfs_ord_config_12['relationship'].to_numpy().astype(int)
                marital_status = cfs_ord_config_12['marital-status'].to_numpy().astype(int)
                sex = cfs_ord_config_12['sex'].to_numpy().astype(int)
                relationship_preserved_config_12 = 0
                for n in range(n_cf):
                    if relationship[n] == 0:
                        relationship_preserved_config_12 += 1 if sex[n]==1 and marital_status[n]==2  else 0
                    elif relationship[n] == 5:
                        relationship_preserved_config_12 += 1 if sex[n]==0 and marital_status[n]==2  else 0
                    else:
                        relationship_preserved_config_12 += 1
                relationship_preserved_config_12 = relationship_preserved_config_12 / n_cf

                # evaluating and recovering counterfactuals of {validity, soundness, coherency} config
                cfs_ord_config_123, \
                cfs_eval_config_123, \
                x_cfs_ord_config_123, \
                x_cfs_eval_config_123 = evaluateCounterfactuals(x_ord, explanation_config_123['cfs_ord'],
                                                                 dataset, predict_fn, predict_proba_fn, task,
                                                                 toolbox, objective_names, featureScaler,
                                                                 feature_names)

                # Education correlation set
                education_num = cfs_ord_config_123['education-num'].to_numpy().astype(int)
                education = cfs_ord_config_123['education'].to_numpy().astype(int)
                education_preserved_config_123 = 0
                for n in range(n_cf):
                    education_preserved_config_123 += 1 if correlations[education_num[n]][1] == education[n] else 0
                education_preserved_config_123 = education_preserved_config_123 / n_cf

                # Relationship correlation set
                relationship = cfs_ord_config_123['relationship'].to_numpy().astype(int)
                marital_status = cfs_ord_config_123['marital-status'].to_numpy().astype(int)
                sex = cfs_ord_config_123['sex'].to_numpy().astype(int)
                relationship_preserved_config_123 = 0
                for n in range(n_cf):
                    if relationship[n] == 0:
                        relationship_preserved_config_123 += 1 if sex[n]==1 and marital_status[n]==2  else 0
                    elif relationship[n] == 5:
                        relationship_preserved_config_123 += 1 if sex[n]==0 and marital_status[n]==2  else 0
                    else:
                        relationship_preserved_config_123 += 1
                relationship_preserved_config_123 = relationship_preserved_config_123 / n_cf

                # evaluating and recovering counterfactuals of {validity, soundness, coherency, actionability} config
                cfs_ord_config_1234, \
                cfs_eval_config_1234, \
                x_cfs_ord_config_1234, \
                x_cfs_eval_config_1234 = evaluateCounterfactuals(x_ord, explanation_config_1234['cfs_ord'],
                                                                 dataset, predict_fn, predict_proba_fn, task,
                                                                 toolbox, objective_names, featureScaler,
                                                                 feature_names)

                # Education correlation set
                education_num = cfs_ord_config_1234['education-num'].to_numpy().astype(int)
                education = cfs_ord_config_1234['education'].to_numpy().astype(int)
                education_preserved_config_1234 = 0
                for n in range(n_cf):
                    education_preserved_config_1234 += 1 if correlations[education_num[n]][1] == education[n] else 0
                education_preserved_config_1234 = education_preserved_config_1234 / n_cf

                # Relationship correlation set
                relationship = cfs_ord_config_1234['relationship'].to_numpy().astype(int)
                marital_status = cfs_ord_config_1234['marital-status'].to_numpy().astype(int)
                sex = cfs_ord_config_1234['sex'].to_numpy().astype(int)
                relationship_preserved_config_1234 = 0
                for n in range(n_cf):
                    if relationship[n] == 0:
                        relationship_preserved_config_1234 += 1 if sex[n]==1 and marital_status[n]==2  else 0
                    elif relationship[n] == 5:
                        relationship_preserved_config_1234 += 1 if sex[n]==0 and marital_status[n]==2  else 0
                    else:
                        relationship_preserved_config_1234 += 1
                relationship_preserved_config_1234 = relationship_preserved_config_1234 / n_cf


                print('\n')
                print('-------------------------------')
                print("%s | %s: %d/%d explained" % (dataset['name'], blackbox_name, explained, N))

                print('\n')
                print(cfs_ord_config_1)
                print(cfs_ord_config_12)
                print(cfs_ord_config_123)
                print(cfs_ord_config_1234)

                print('\n')
                print("Preserved Education    coherency | Validity: %0.3f - Validity+Soundness: %0.3f - "
                      "Validity+Soundness+Coherency: %0.3f - Validity+Soundness+Coherency+Actionability: %0.3f" %
                      (education_preserved_config_1, education_preserved_config_12,
                       education_preserved_config_123, education_preserved_config_1234))
                print("Preserved Relationship coherency | Validity: %0.3f - Validity+Soundness: %0.3f - "
                      "Validity+Soundness+Coherency: %0.3f - Validity+Soundness+Coherency+Actionability: %0.3f" %
                      (relationship_preserved_config_1, relationship_preserved_config_12,
                       relationship_preserved_config_123, relationship_preserved_config_1234))
                print('-------------------------------------------------------------------------------------'
                      '-------------------------------------------------------------------------------------')

                # storing the evaluation of the best counterfactual found by methods
                eval_results = np.r_[education_preserved_config_1, education_preserved_config_12,
                                     education_preserved_config_123, education_preserved_config_1234,
                                     relationship_preserved_config_1, relationship_preserved_config_12,
                                     relationship_preserved_config_123, relationship_preserved_config_1234]
                eval_results = ['%.3f' % (eval_results[i]) for i in range(len(eval_results))]
                eval_results = ','.join(eval_results)
                eval_results_csv.write('%s\n' % (eval_results))
                eval_results_csv.flush()

                explained += 1

                if explained == N:
                    break

            eval_results_csv.close()

if __name__ == '__main__':
    main()
