import torch
torch.set_num_threads(10)
torch.manual_seed(0)
import torch.nn as nn
from torch.autograd import Variable
from tensorflow import keras
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def ModelConstruction(X_train, X_test, Y_train, Y_test, model_name, constructor):

    # constructing the black-box model
    if model_name == 'nn':
        class DNN(nn.Module):

            # model's architecture
            def __init__(self, n_inputs):
                super(DNN, self).__init__()
                # input to first hidden layer
                self.hidden1 = Linear(n_inputs, 500)
                kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
                self.act1 = ReLU()
                # second hidden layer
                self.hidden2 = Linear(500, 500)
                kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
                self.act2 = ReLU()
                # third hidden layer and output
                self.hidden3 = Linear(500, 2)
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

            # predict label
            def predict(self, x):
                with torch.no_grad():
                    output = self.forward(x)
                    _, label = torch.max(output.data, 1)
                    label = label.cpu().detach().numpy()
                    return label

            # predict proba
            def predict_proba(self, x):
                with torch.no_grad():
                    output = self.forward(x)
                    proba = output.cpu().numpy()
                    return proba

        # training the model
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
                _, indices = torch.max(output, 1)
                preds = torch.tensor(keras.utils.to_categorical(indices, num_classes=n_classes))
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                current_loss += loss.item()
                current_correct += (preds.int() == labels.int()).sum() / n_classes
            current_loss = current_loss / X.size(0)
            current_correct = current_correct.double() / X.size(0)
            return preds, current_loss, current_correct.item()

        # data preparation
        n_classes = len(np.unique(Y_train))
        x_train = torch.FloatTensor(X_train)
        y_train = keras.utils.to_categorical(Y_train, n_classes)
        y_train = torch.FloatTensor(y_train)

        # training hyper-parameteres
        epochs = 100
        batch_size = 128
        blackbox = DNN(X_train.shape[1])
        lr = 1e-4

        # training the model
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(blackbox.parameters(), lr=lr)
        for epoch in range(epochs):
            preds, epoch_loss, epoch_acc = train(blackbox, criterion, optimizer, x_train, y_train, batch_size, n_classes)
            # print("> epoch {:.0f}\tLoss {:.5f}\tAcc {:.5f}".format(epoch, epoch_loss, epoch_acc))

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
    else:
        pred_train = blackbox.predict(X_train)


    acc_train = np.round(accuracy_score(Y_train, pred_train),3)
    print(model_name, '| Train data | Accuracy      =', acc_train)

    precision_train = np.round(precision_score(Y_train, pred_train, average='weighted'),3)
    print(model_name, '| Train data | Precision     =', precision_train)

    recall_train = np.round(recall_score(Y_train, pred_train, average='weighted'),3)
    print(model_name, '| Train data | Recall        =', recall_train)

    f1_train = np.round(f1_score(Y_train, pred_train, average='weighted'),3)
    print(model_name, '| Train data | F1-score      =', f1_train)

    roc_auc_train = np.round(roc_auc_score(Y_train, pred_train, average='weighted'),3)
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

    precision_test  = np.round(precision_score(Y_test, pred_test, average='weighted'),3)
    print(model_name, '| Test data | Precision     =', precision_test)

    recall_test  = np.round(recall_score(Y_test, pred_test, average='weighted'),3)
    print(model_name, '| Test data | Recall        =', recall_test)

    f1_test  = np.round(f1_score(Y_test, pred_test, average='weighted'),3)
    print(model_name, '| Test data | F1-score      =', f1_test)

    roc_auc_test  = np.round(roc_auc_score(Y_test, pred_test, average='weighted'),3)
    print(model_name, '| Test data | ROC-AUC score =', roc_auc_test)

    test_performance = {'acc': acc_test,
                        'precision':precision_test,
                        'recall':recall_test,
                        'f1':f1_test,
                        'roc_auc':roc_auc_test}
    print('\n')

    return blackbox, train_performance, test_performance

