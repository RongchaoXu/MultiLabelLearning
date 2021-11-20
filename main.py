from sklearn.svm import SVC
import pandas as pd
from dataset import get_dataset
from sklearn.metrics import accuracy_score


def computeACC(pred, gt):
    intersection = 0
    union = 0
    for i in range(len(pred)):
        sum = pred[i] + gt[i]
        if sum == 2:
            intersection += 1
        if sum > 0:
            union += 1
    return intersection / union


if __name__ == '__main__':
    x_train = './Data for Problem 1/X_train.mat'
    y_train = './Data for Problem 1/y_train.mat'
    x_test = './Data for Problem 1/X_test.mat'
    y_test = './Data for Problem 1/y_test.mat'
    kernel = 'poly'  # 'rbf' / 'poly'
    c = 2

    x_train_pd, y_train_pd, x_test_pd, y_test_pd = get_dataset(x_train, x_test, y_train, y_test)
    models = []
    for i in range(6):
        models.append(SVC(C=c, kernel=kernel))
    # train
    for i in range(6):
        models[i].fit(x_train_pd, y_train_pd[i])

    # test
    acc = []
    for i in range(6):
        result = []
        pred_labels = models[i].predict(x_test_pd)
        acc.append(computeACC(pred_labels, y_test_pd[i].to_numpy()))
    print(sum(acc) / 6)
