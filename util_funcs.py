import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.neighbors import NearestNeighbors
from LowProFool.Adverse import lowProFool, deepfool

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


# evaluating the performance of baseline methods
def evaluate_performance(config):

    TrainData = config['TrainData']
    X_KNN = TrainData.values[:,:-1]
    Y_KNN = TrainData.values[:, -1].astype(int)

    KNN = [[],[]]
    for j in [0,1]:
        ind_j = np.where(Y_KNN==j)[0]
        X_j = X_KNN[ind_j,:]
        Y_j = Y_KNN[ind_j]
        KNN[j] = NearestNeighbors(n_neighbors=50).fit(X_j, Y_j)

    performance_all = {}
    for method, results in config['AdvData'].items():

        X_orig = np.asarray(results['X_orig'])
        X_adv = np.asarray(results['X_adv'])
        predict_fn = config['Predict_fn']
        predict_proba_fn =  config['Predict_proba_fn']

        pred_orig = predict_fn(X_orig)
        pred_adv =predict_fn(X_adv)

        validity = (pred_orig != pred_adv)

        perturbations = np.abs(X_orig - X_adv)
        norm_perturbations =  np.linalg.norm(perturbations, axis=1)

        robustness = 0
        for j in [0,1]:
            ind = np.where(pred_orig==j)[0]
            D_j = norm_perturbations[ind]
            N_j = len(ind)
            robustness = robustness + (sum(D_j)/N_j)

        target_class = 1 - predict_fn(X_orig)
        proba_adv = predict_proba_fn(X_adv)
        proba_adv = [proba_adv[i, t] for i, t in enumerate(target_class)]

        dist_to_nn = []
        dist_to_knn = []
        for x, c in zip(X_adv, target_class):
            distances, indices = KNN[c].kneighbors(x.reshape(1,-1))
            dist_to_nn.append(distances[0][0])
            dist_to_knn.append(np.mean(distances[0]))

        exe_time = np.asarray(results['Time'])

        overall_robustness = [np.sum(validity)/len(validity), np.round(robustness,3),
                              np.round(np.mean(norm_perturbations),3), np.round(np.std(norm_perturbations),3),
                              np.round(np.mean(proba_adv),3),  np.round(np.std(proba_adv),3),
                              np.round(np.mean(dist_to_nn),3),  np.round(np.std(dist_to_nn),3),
                              np.round(np.mean(dist_to_knn), 3), np.round(np.std(dist_to_knn),3),
                              np.round(np.mean(exe_time), 3), np.round(np.std(exe_time), 3)]


        class_robustness = [[], []]
        for j in [0,1]:

            validity_j = validity[np.where(pred_orig==j)]

            ind_i = np.where(pred_orig == (1-j))[0]
            D_i = norm_perturbations[ind_i]
            N_i = len(ind_i)
            alpha = np.sum(D_i) / N_i

            ind_j = np.where(pred_orig == j)[0]
            D_j = norm_perturbations[ind_j]
            N_j = len(ind_j)
            beta = np.sum(D_j) / N_j

            robustness_j = alpha / beta

            norm_perturbations_j = norm_perturbations[ind_j]

            proba_adv_j = np.asarray(proba_adv)[ind_j]

            dist_to_nn_j = np.asarray(dist_to_nn)[ind_j]

            dist_to_knn_j = np.asarray(dist_to_knn)[ind_j]

            exe_time_j = exe_time[ind_j]

            robustness_j = [np.sum(validity_j)/len(validity_j), np.round(robustness_j, 3),
                            np.round(np.mean(norm_perturbations_j), 3), np.round(np.std(norm_perturbations_j), 3),
                            np.round(np.mean(proba_adv_j), 3), np.round(np.std(proba_adv_j), 3),
                            np.round(np.mean(dist_to_nn_j), 3), np.round(np.std(dist_to_nn_j), 3),
                            np.round(np.mean(dist_to_knn_j), 3), np.round(np.std(dist_to_knn_j), 3),
                            np.round(np.mean(exe_time_j), 3), np.round(np.std(exe_time_j), 3)]

            class_robustness[j] = robustness_j

        robustness = []
        robustness.append(overall_robustness)
        for l in class_robustness:
            robustness.append(l)
        robustness = np.asarray(robustness)

        performance = pd.DataFrame(columns=['SuccessRate', 'Robustness',
                                            'MeanNormPerturbation', 'StdNormPerturbation',
                                            'MeanProbability', 'StdProbability',
                                            'MeanNearestNeighbor', 'StdNearestNeighbor',
                                            'MeanKNNN', 'StdKNN',
                                            'MeanTime', 'StdTime'], data=robustness,
                                   index=['Overall', 'Class_0', 'Class_1'])

        performance_all[method] = performance

    return performance_all