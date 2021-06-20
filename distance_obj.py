from sklearn.metrics import pairwise_distances

def distanceObj(x_ord, cf_ord, feature_width, continuous_indices, discrete_indices):
    distance = pairwise_distances(x_ord.reshape(1,-1), cf_ord.reshape(1,-1),
                                  metric='minkowski', p=2)[0][0]
    return distance

# import numpy as np
#
# def distanceObj(x_ord, cf_ord, feature_width, continuous_indices, discrete_indices):
#     distance = []
#     if continuous_indices is not None:
#         for j in continuous_indices:
#             distance.append((1/feature_width[j]) * abs(x_ord[j]-cf_ord[j]))
#     if discrete_indices is not None:
#         for j in discrete_indices:
#             distance.append(int(x_ord[j] != cf_ord[j]))
#     return np.mean(distance)