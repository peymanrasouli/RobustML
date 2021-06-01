import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from tensorflow import keras
# pytorch mlp for binary classification
from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def ModelConstruction(X_train, X_test, Y_train, Y_test, model_name, constructor):

    # constructing the black-box model
    if model_name == 'nn':
        class DNN(nn.Module):
            def __init__(self, n_inputs):
                super(DNN, self).__init__()
                # input to first hidden layer
                self.hidden1 = Linear(n_inputs, 50)
                kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
                self.act1 = ReLU()
                # second hidden layer
                self.hidden2 = Linear(50, 50)
                kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
                self.act2 = ReLU()
                # third hidden layer and output
                self.hidden3 = Linear(50, 2)
                xavier_uniform_(self.hidden3.weight)
                self.act3 = Sigmoid()

                # forward propagate input

            def forward(self, X):
                # input to first hidden layer
                X = self.hidden1(X)
                X = self.act1(X)
                # second hidden layer
                X = self.hidden2(X)
                X = self.act2(X)
                # third hidden layer and output
                X = self.hidden3(X)
                X = self.act3(X)
                return X

            # def __init__(self, D_in, H, D_out):
            #     super(DNN, self).__init__()
            #     self.linear1 = torch.nn.Linear(D_in, H)
            #     self.linear2 = torch.nn.Linear(H, H)
            #     self.linear3 = torch.nn.Linear(H, H)
            #     self.linear4 = torch.nn.Linear(H, D_out)
            #     self.relu = torch.nn.ReLU()
            #     self.softmax = torch.nn.Softmax(dim=0)
            #
            # def forward(self, x):
            #     h1 = self.relu(self.linear1(x))
            #     h2 = self.relu(self.linear2(h1))
            #     h3 = self.relu(self.linear3(h2))
            #     a3 = self.linear4(h3)
            #     y = self.softmax(a3)
            #     return y



            def predict(self, x):
                with torch.no_grad():
                    output = self.forward(x)
                    _, label = torch.max(output.data, 1)
                    label = label.cpu().detach().numpy()
                    return label

            def predict_proba(self, x):
                with torch.no_grad():
                    output = self.forward(x)
                    proba = output.cpu().numpy()
                    return proba

        def train(model, criterion, optimizer, X, y, N, n_classes):
            model.train()
            current_loss = 0
            current_correct = 0
            # Training in batches
            for ind in range(0, X.size(0), N):
                indices = range(ind, min(ind + N, X.size(0)) - 1)
                inputs, labels = X[indices], y[indices]
                inputs = Variable(inputs, requires_grad=True)
                optimizer.zero_grad()
                output = model(inputs)
                _, indices = torch.max(output, 1)  # argmax of output [[0.61, 0.12]] -> [0]
                # [[0, 1, 1, 0, 1, 0, 0]] -> [[1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0], [1, 0]]
                preds = torch.tensor(keras.utils.to_categorical(indices, num_classes=n_classes))
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                current_loss += loss.item()
                current_correct += (preds.int() == labels.int()).sum() / n_classes
            current_loss = current_loss / X.size(0)
            current_correct = current_correct.double() / X.size(0)
            return preds, current_loss, current_correct.item()

        n_classes = len(np.unique(Y_train))
        x_train = torch.FloatTensor(X_train)
        y_train = keras.utils.to_categorical(Y_train, n_classes)
        y_train = torch.FloatTensor(y_train)

        D_in = x_train.size(1)
        D_out = y_train.size(1)
        epochs = 100
        batch_size = 100
        H = 50
        blackbox = DNN(X_train.shape[1])
        lr = 1e-4

        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(blackbox.parameters(), lr=lr)
        for epoch in range(epochs):
            preds, epoch_loss, epoch_acc = train(blackbox, criterion, optimizer, x_train, y_train, batch_size, n_classes)
            print("> epoch {:.0f}\tLoss {:.5f}\tAcc {:.5f}".format(epoch, epoch_loss, epoch_acc))

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
        pred_train = blackbox.predict((Variable(torch.tensor(X_train).float())))
        # proba_train = blackbox.predict_proba((Variable(torch.tensor(X_train).float())))
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
        pred_test = blackbox.predict((Variable(torch.tensor(X_test).float())))
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

