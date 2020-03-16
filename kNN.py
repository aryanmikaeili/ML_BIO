from utils import *
import numpy as np
class kNN:
    def __init__(self, k):
        self.k = k

    def fit(self, x, y):
        self.x = x.values
        self.y = np.array(y)
        self.factorized_labels = pd.factorize(self.y)
        self.label_size = len(self.factorized_labels[1])
        return

    def predict(self, x):
        dists = []
        for i in range(len(self.x)):
            d = (x - np.array(self.x[i]))
            dists.append([i, np.linalg.norm(d)])

        dists = sorted(dists, key=lambda row:row[1])
        dists = np.array(dists, dtype=int)[:self.k, 0]
        label_scores = [0 for i in range(self.label_size)]
        for i in range(self.k):
            label_scores[self.factorized_labels[0][dists[i]]] += 1

        correct_factor = np.argmax(label_scores)
        correct_label = self.factorized_labels[1][correct_factor]
        return correct_label


        a = 0

    def test(self, test_x):
        preds = []
        for i in range(len(test_x)):
            p = self.predict(test_x.iloc[i])
            preds.append(p)
        return preds






train_x, train_y, test_x, test_y = read_and_prepare("heart.csv","target", 0.8)


max_k = 15




