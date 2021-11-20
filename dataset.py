import pandas as pd
from pandas import DataFrame
import scipy.io
import os
import re


def get_df(matrix_path, label_path, key1, key2):
    matrix_set = scipy.io.loadmat(matrix_path)[key1]
    label_set = scipy.io.loadmat(label_path)[key2]
    matrix_df = pd.DataFrame(matrix_set)
    label_df = pd.DataFrame(label_set)
    return matrix_df, label_df


def split_label(test, num=6):
    result = []
    for i in range(num):
        result.append(test.iloc[:, [i]])
    return result


def get_dataset(x_train_path, x_test_path, y_train_path, y_test_path):
    x_train_pd, y_train_pd = get_df(x_train_path, y_train_path, 'X_train', 'y_train')
    x_test_pd, y_test_pd = get_df(x_test_path, y_test_path, 'X_test', 'y_test')
    y_test_pd = split_label(y_test_pd)
    y_train_pd = split_label(y_train_pd)
    return x_train_pd, y_train_pd, x_test_pd, y_test_pd


if __name__ == '__main__':
    x_train_pd, y_train_pd, x_test_pd, y_test_pd = get_dataset('./Data for Problem 1/X_train.mat', './Data for Problem 1/X_test.mat',
                './Data for Problem 1/y_train.mat', './Data for Problem 1/y_test.mat')
    print(x_test_pd.shape, x_train_pd.shape)
    print(y_train_pd)