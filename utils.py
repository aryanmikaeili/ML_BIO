import pandas as pd
import numpy as np
import math
import os

class report:
    def __init__(self, acc, perc, rec, spec, f1):
        self.acc = acc
        self.perc = perc
        self.rec = rec
        self.spec = spec
        self.f1 = f1

    def print_report(self):
        print("accuracy: ", self.acc)
        print("percision: ", self.perc)
        print("recall: ", self.rec)
        print("specificity: ", self.spec)
        print("f1-score: ", self.f1)


def data_read(name, target_name):
    address = os.getcwd() + "\\" + name
    data = pd.read_csv(address)
    return data.drop(target_name, axis=1), data[target_name]


def shuffle(x, y):
    size = len(x)
    shuffled_x = pd.DataFrame(columns=list(x.columns))
    shuffled_y = pd.Series([0 for i in range(size)])
    random_indexes = np.random.permutation(size)
    for i in range(size):
        shuffled_x.loc[i] = x.iloc[random_indexes[i]]
        shuffled_y[i] = y[random_indexes[i]]

    return shuffled_x, shuffled_y


def split_data(x, y, f):
    size = len(x)
    training_size = math.floor(f * size)
    trainin_x = x.iloc[:training_size]
    trainin_y = y[:training_size]
    test_x = x.iloc[training_size:]
    test_y = y[training_size:]

    return trainin_x, trainin_y, test_x, test_y



##all in one
def read_and_prepare(name, target_name, split_frac):
    data_x, data_y = data_read("heart.csv", target_name)
    data_x_shuffled, data_y_shuffled = shuffle(data_x, data_y)
    train_x, train_y, test_x, test_y = split_data(data_x_shuffled, data_y_shuffled, split_frac)
    return train_x, train_y, test_x, test_y

def accuracy(predictions, labels):
    number_of_true =  np.sum(predictions == labels)
    return (number_of_true / len(predictions)) * 100

def confusion_matrix(prediction, labels):
    true_preds = np.array(prediction)[np.where(prediction == labels)]
    one_predictions = np.sum(prediction)
    TP = np.sum(true_preds)
    TN = len(true_preds) - TP
    FP = one_predictions - TP
    FN = len(labels) - one_predictions - TN

    return [[TP, FP], [FN, TN]]

def classification_report(prediction, label):
    cm = confusion_matrix(prediction, label)
    TP = cm[0][0]
    TN = cm[1][1]
    FP = cm[0][1]
    FN = cm[1][0]
    percision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * ((percision * recall) / (percision + recall))
    spec = TN / (TN + FP)
    acc = accuracy(prediction, label)
    rep = report(acc, percision, recall, spec, f1_score)
    return rep

def make_cross_val(x, y, d):

    size = len(x)
    val_x = []
    val_y = []
    for i in range(d):
        st = round((size / d) * i)
        en = round((size / d) * (i + 1))
        val_x.append([x[st:en], x.drop(x.index[st:en], inplace=False)])
        val_y.append([y[st:en], y.drop(y.index[st:en], inplace=False)])

    return val_x, val_y

def t_test(preds_x, preds_y):
    mean_x = np.mean(preds_x)
    mean_y = np.mean(preds_y)
    temp_x = preds_x - mean_x
    temp_y = preds_y - mean_y
    temp_x = np.sum(np.power(temp_x, 2))
    temp_y = np.sum(np.power(temp_y, 2))
    s_x = temp_x /(len(preds_x) - 1)
    s_y = temp_y / (len(preds_y) - 1)

    t = (mean_x - mean_y) / (np.sqrt((s_x / len(preds_x)) + (s_y / len(preds_y))))

    return t




train_x, train_y, test_x, test_y = read_and_prepare("heart.csv","target", 0.8)

make_cross_val(train_x, train_y, 5)
t_test(test_y[0:10].values, test_y[10:20].values)

