from utils import *
import numpy as np
import math

class node:
    def __init__(self, data, label, category, feature, is_cat):
        self.data = data
        self.feature = feature
        self.label = label
        self.children = []
        self.cat = category
        self.is_cat = is_cat


class decision_tree:
    def __init__(self, max_depth, threshold):
        self.max_depth = max_depth
        self.threshold = threshold
        self.root = None

    def fit(self, x, y):

        self.is_c = self.is_categorical(x)
        root = node(x, y, 0, "", True)
        self.root = root
        tree = self.construct_tree(root, x.columns, 1, self.is_c)
        return tree

    def predict(self, x):
        current = self.root
        while True:
            if len(current.children) == 0:
                cat, count = np.unique(current.label, return_counts=True)
                if len(count) == 2:
                    if count[0] > count[1]:
                        return 0
                    return 1
                else:
                    return cat[0]

            current_feature = x[current.feature]

            if current.is_cat:
                for i in current.children:
                    if i.cat == current_feature:
                        current = i
                        break
            else:
                if current_feature < current.children[0].cat:
                    current = current.children[0]
                else:
                    current = current.children[1]






    def split_data(self, data, label, column, threshold):
        splitting = data[column]
        below_x = data[splitting <= threshold]
        below_y = label[splitting <= threshold]

        above_x = data[splitting > threshold]
        above_y = label[splitting > threshold]

        return below_x, below_y, above_x, above_y
    def entropy(self, labels):
        _, counts = np.unique(labels, return_counts=True)
        size = len(labels)
        probs = counts / size
        entropy = -1 * np.sum(np.log2(probs) * probs)
        return entropy
    def cond_entropy(self, below, above):
        entropy_below = self.entropy(below)
        entropy_above = self.entropy(above)
        p_below = len(below) / (len(below) + len(above))
        p_above = len(above) / (len(below) + len(above))
        return p_below * entropy_below + p_above * entropy_above

    def cond_entropy_categorical(self, data, label, column):
        current = data[column]
        cats, counts = np.unique(current, return_counts=True)
        size = len(current)
        probs = counts / size
        cat_data = []
        cat_y = []
        for i in range(len(cats)):
            cat_data.append(data[current == cats[i]])
            cat_y.append(label[current == cats[i]])

        entropy = 0
        for i in range(len(cats)):
            e = self.entropy(cat_y[i])
            entropy += e * probs[i]

        return entropy

    def choose_feature(self, data, label, columns, is_categorical):
        min_entropy = math.inf
        best_feature = 0
        is_cat = True
        for i in range(len(columns)):
            if is_categorical[i]:
                entropy = self.cond_entropy_categorical(data, label, columns[i])
                if entropy < min_entropy:
                    min_entropy = entropy
                    best_feature = columns[i]
                    is_cat = True
            else:
                entropy, split = self.best_split(data, label, columns[i])
                if entropy < min_entropy:
                    min_entropy = entropy
                    best_feature = columns[i]
                    is_cat = False

        return best_feature, is_cat

    def is_categorical(self, data):
        is_cat = []
        for i in range(len(data.columns)):
            if len(np.unique(data[data.columns[i]])) >= 4:
                is_cat.append(False)
            else:
                is_cat.append(True)
        return is_cat

    def construct_tree(self, tree, features, depth, is_categorical):
        if len(features) == 0:
            return tree
        elif depth == self.max_depth:
            return tree
        elif (np.sum(tree.label) / len(tree.label)) >= self.threshold or (np.sum(tree.label) / len(tree.label)) <= 1 - self.threshold:
            return tree

        best_feature, is_cat = self.choose_feature(tree.data, tree.label, features, is_categorical)
        tree.feature = best_feature
        tree.is_cat = is_cat
        if  best_feature == "ca":
            a = 0
        if is_cat:
            current = tree.data[best_feature]
            cats, counts = np.unique(current, return_counts=True)
            cat_data = []
            cat_y = []
            for i in range(len(cats)):
                cat_data.append(tree.data[current == cats[i]])
                cat_y.append(tree.label[current == cats[i]])

            for i in range(len(cats)):
                new_node = node(cat_data[i], cat_y[i], cats[i], "", True)
                tree.children.append(new_node)
                self.construct_tree(new_node, np.delete(features, np.argwhere(features==best_feature)), depth + 1, np.delete(is_categorical, np.argwhere(features==best_feature)))
        else:
            _, split = self.best_split(tree.data, tree.label, best_feature)
            below_x, below_y, above_x, above_y = self.split_data(tree.data, tree.label,best_feature, split)

            new_node_below = node(below_x, below_y, split, "", True)
            new_node_above = node(above_x, above_y, split, "", True)

            tree.children.append(new_node_below)
            tree.children.append(new_node_above)

            self.construct_tree(new_node_below, features, depth + 1, is_categorical)
            self.construct_tree(new_node_below, features, depth + 1, is_categorical)








    def best_split(self, data, label, column):

        splitting = data[column]
        unique_values = np.unique(splitting)
        min_entropy = math.inf
        best_split = 0
        for i in range(1, len(unique_values)):
            splitting_value = (unique_values[i - 1] + unique_values[i]) / 2
            below_x, below_y, above_x, above_y = self.split_data(data, label, column, splitting_value)
            entropy = self.cond_entropy(below_y, above_y)
            if entropy < min_entropy:
                min_entropy = entropy
                best_split = splitting_value


        return min_entropy, best_split
    def test(self, test_x):
        preds = []
        for i in range(len(test_x)):
            p = self.predict(test_x.iloc[i])
            preds.append(p)
        return preds

















