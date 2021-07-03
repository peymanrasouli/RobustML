import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_circles,make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

h = .005  # step size in the mesh

classifiers = []
names = []

classifiers.append(make_pipeline(
    StandardScaler(),
    MLPClassifier(
        solver='lbfgs', alpha=0.3, random_state=1, max_iter=2000,
        early_stopping=True, hidden_layer_sizes=[100, 100],
    )
    ))

names.append("Narrow Margin")


classifiers.append(make_pipeline(
    StandardScaler(),
    MLPClassifier(
        solver='lbfgs', alpha=1, random_state=1, max_iter=2000,
        early_stopping=True, hidden_layer_sizes=[100, 100],
    )
    ))
names.append("Large Margin")

# # creating circles data set
# X, y = make_circles(n_samples=200, noise=0.2, factor=0.5)

X, y = make_classification(n_features=2, n_samples=100, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)

# split into training and test part
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

figure = plt.figure(figsize=(6,3))
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#a079b0', '#72bb76'])

# iterate over classifiers
i = 1
for name, clf in zip(names, classifiers):
    ax = plt.subplot(1, 2, i)
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
    ax.contourf(xx, yy, Z, cmap=cm, alpha=1.0)

    # Plotting training data point in class 0
    class_0 = ax.scatter(X[np.where(y==0)[0], 0], X[np.where(y==0)[0], 1], c='#FF0000',
               edgecolors='white', s=20, alpha=1.0, label='class 0')

    # Plotting training data point in class 1
    class_1 = ax.scatter(X[np.where(y==1)[0], 0], X[np.where(y==1)[0], 1], c='#0000FF',
                         edgecolors='white', s=20,  alpha=1.0, label='class 1')

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    name_score = name + ' | Accuracy: ' +  str(np.round(score,2))
    ax.set_title(name_score)
    ax.legend((class_0, class_1),
               ('class 0', 'class 1'),
               scatterpoints=1,
               loc='upper left',
               ncol=1,
               fontsize=10)
    i+=1

figure.subplots_adjust(left=.02, right=0.98)
plt.show()
figure.savefig('margin_visualization.pdf', bbox_inches = 'tight')
plt.close()