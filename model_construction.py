import torch
from torch.autograd import Variable
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def ModelConstruction(dataset, X_train, X_test, Y_train, Y_test, model_name, constructor):

    # constructing the black-box model
    if model_name == 'nn':
        pass
    elif model_name=='rf' or model_name=='gb':
        # constructing the black-box model
        blackbox = constructor(random_state=42, n_estimators=100)
        # fitting the black-box model
        blackbox.fit(X_train, Y_train)

    elif model_name=='svc':
        # constructing the black-box model
        blackbox = constructor(random_state=42, probability=True)
        # fitting the black-box model
        blackbox.fit(X_train, Y_train)

    else:
        # constructing the black-box model
        blackbox = constructor(random_state=42)
        # fitting the black-box model
        blackbox.fit(X_train, Y_train)


    # model's performance on the train data
    if model_name == 'nn':
        pass
    else:
        pred_train = blackbox.predict(X_train)


    acc_train = np.round(accuracy_score(Y_train, pred_train),3)
    print(model_name, '| Train data | Accuracy      =', acc_train)

    precision_train = np.round(precision_score(Y_train, pred_train, average='macro'),3)
    print(model_name, '| Train data | Precision     =', precision_train)

    recall_train = np.round(recall_score(Y_train, pred_train, average='macro'),3)
    print(model_name, '| Train data | Recall        =', recall_train)

    f1_train = np.round(f1_score(Y_train, pred_train, average='macro'),3)
    print(model_name, '| Train data | F1-score      =', f1_train)

    roc_auc_train = np.round(roc_auc_score(Y_train, pred_train, average='macro'),3)
    print(model_name, '| Train data | ROC-AUC score =', roc_auc_train)

    train_performance = {'acc': acc_train,
                        'precision': precision_train,
                        'recall': recall_train,
                        'f1': f1_train,
                        'roc_auc': roc_auc_train}
    print('\n')


    # model's performance on the test data
    if model_name == 'nn':
        pass
    else:
        pred_test = blackbox.predict(X_test)

    acc_test = np.round(accuracy_score(Y_test, pred_test),3)
    print(model_name, '| Test data | Accuracy      =', acc_test)

    precision_test  = np.round(precision_score(Y_test, pred_test, average='macro'),3)
    print(model_name, '| Test data | Precision     =', precision_test)

    recall_test  = np.round(recall_score(Y_test, pred_test, average='macro'),3)
    print(model_name, '| Test data | Recall        =', recall_test)

    f1_test  = np.round(f1_score(Y_test, pred_test, average='macro'),3)
    print(model_name, '| Test data | F1-score      =', f1_test)

    roc_auc_test  = np.round(roc_auc_score(Y_test, pred_test, average='macro'),3)
    print(model_name, '| Test data | ROC-AUC score =', roc_auc_test)

    test_performance = {'acc': acc_test,
                        'precision':precision_test,
                        'recall':recall_test,
                        'f1':f1_test,
                        'roc_auc':roc_auc_test}
    print('\n')

    return blackbox, train_performance, test_performance

