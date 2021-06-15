print(__doc__)


# Author: Issam H. Laradji
# License: BSD 3 clause

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

h = .02  # step size in the mesh

classifiers = []
names = []

classifiers.append(make_pipeline(
    StandardScaler(),
    MLPClassifier(
        solver='lbfgs', alpha=0.1, random_state=1, max_iter=2000,
        early_stopping=True, hidden_layer_sizes=[100, 100],
    )
    ))

names.append("Small Margin")


classifiers.append(make_pipeline(
    StandardScaler(),
    MLPClassifier(
        solver='lbfgs', alpha=2, random_state=1, max_iter=2000,
        early_stopping=True, hidden_layer_sizes=[100, 100],
    )
    ))
names.append("Large Margin")


datasets = [
            make_circles(n_samples=200, noise=0.2, factor=0.5),
            ]

figure = plt.figure(figsize=(6,3))
i = 1
# iterate over datasets
for X, y in datasets:
    # split into training and test part
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers), i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max] x [y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='black', s=25, label='Class 0')
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   alpha=0.6, edgecolors='black', s=25, label='Class 1')

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        name_score = name + ' | Accuracy: ' +  str(np.round(score,2))
        ax.set_title(name_score)
        ax.legend(loc='upper right')
        i += 1

figure.subplots_adjust(left=.02, right=0.98)
plt.show()
figure.savefig('margin_visualization.pdf', bbox_inches = 'tight')
plt.close()