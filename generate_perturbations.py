import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from LowProFool.Adverse import lowProFool, deepfool
from console_progressbar.progressbar import ProgressBar

def calculateWeights(df, target, show_heatmap=False):
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

def generatePerturbations(config, method):
    df_test = config['TestData']
    model = config['Model']
    weights = config['Weights']
    bounds = config['Bounds']
    maxiters = config['MaxIters']
    alpha = config['Alpha']
    lambda_ = config['Lambda']
    feature_names = config['FeatureNames']
    MOCE = config['MOCE']

    results = {'X_orig': [],
               'X_adv': [],
               'Time': []}

    pb = ProgressBar(total=len(df_test), prefix='Progress:', suffix='Complete',
                     decimals=1, length=50, fill='█', zfill='-')
    i = 0
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
        elif method == 'MOCE':
            start_time = time.time()
            explanation = MOCE.explain(x_orig)
            x_adv = explanation['best_cf_ord'].to_numpy()
            spent_time = time.time() - start_time
        else:
            raise Exception("Invalid method", method)

        results['X_orig'].append(x_orig)
        results['X_adv'].append(x_adv)
        results['Time'].append(spent_time)

        i += 1
        pb.print_progress_bar(i)

    return results


