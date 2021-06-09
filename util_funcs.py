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
        exe_time = np.asarray(results['Time'])
        predict_fn = config['Predict_fn']
        predict_proba_fn =  config['Predict_proba_fn']

        # calculate the metrics based on valid examples
        pred_orig = predict_fn(X_orig)
        pred_adv =predict_fn(X_adv)
        ind_valid = np.where(pred_orig != pred_adv)[0]
        X_orig = X_orig[ind_valid]
        X_adv = X_adv[ind_valid]
        exe_time = exe_time[ind_valid]

        # method's name
        metrics.append(method)

        # success rate
        success_rate = np.round((len(ind_valid)/np.asarray(results['X_orig']).shape[0]),3)
        metrics.append(success_rate)

        # norm perturbations
        perturbations = np.abs(X_orig - X_adv)
        norm_perturbations =  np.linalg.norm(perturbations, axis=1)
        mean_norm_perturbations = np.round(np.mean(norm_perturbations),3)
        std_norm_perturbations = np.round(np.std(norm_perturbations),3)
        metrics.append(mean_norm_perturbations)
        metrics.append(std_norm_perturbations)

        # probability
        target_class = 1 - predict_fn(X_orig)
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
        mean_exe_time = np.round(np.mean(exe_time),3)
        std_exe_time = np.round(np.std(exe_time),3)
        metrics.append(mean_exe_time)
        metrics.append(std_exe_time)

        performance = performance.append(pd.DataFrame(columns=performance.columns,
                                                      data=np.asarray(metrics).reshape(1,-1)),
                                         ignore_index=True)
    return performance