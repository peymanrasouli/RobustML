import numpy as np

def nonboundaryProbabilityObj(cf_ord, predict_proba_fn, probability_thresh, cf_class):
    cf_probability = predict_proba_fn(cf_ord.reshape(1, -1))[0, cf_class]
    cost = np.max([0, probability_thresh - cf_probability])
    return cost