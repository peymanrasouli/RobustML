import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# evaluating the performance of baseline methods
def evaluatePerformance(config):

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