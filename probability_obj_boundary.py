import numpy as np

def boundaryProbabilityObj(cf_ord, predict_proba_fn, probability_thresh, cf_class):
    cf_probability = predict_proba_fn(cf_ord.reshape(1, -1))[0, cf_class]
    cost = np.abs(probability_thresh - cf_probability)
    return cost