def labelObj(cf, predict_fn, cf_class):
    cf_pred = predict_fn(cf.reshape(1, -1))[0]
    cost = 0.0 if cf_pred==cf_class else 1.0
    return cost
