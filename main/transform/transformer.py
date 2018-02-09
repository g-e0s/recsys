from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.model_selection import train_test_split

from main.data_access_layer.input import Input


class Transformer(metaclass=ABCMeta):
    def __init__(self):
        self.levels = open("items.txt", "r").read().split("\n")


class LearningTransformer(Transformer, metaclass=ABCMeta):
    @abstractmethod
    def split(self, dataset):
        pass

    @abstractmethod
    def transform(self, dataset, func):
        pass


class SciPyLearningTransformer(LearningTransformer):
    def split(self, dataset, period=30):
        period_milliseconds = period * 24 * 60 * 60 * 1000
        orders = dataset[["userID", "timestamp"]].drop_duplicates()
        train, test = train_test_split(orders, train_size=0.8)

        x_train = train \
            .merge(dataset, on="userID", suffixes=("", "_0")) \
            .query("timestamp_0 >= timestamp - {} and timestamp_0 < timestamp".format(period_milliseconds)) \
            .drop("timestamp_0", axis=1)

        y_train = x_train[["userID", "timestamp"]].drop_duplicates().merge(dataset, on=["userID", "timestamp"])

        x_test = test \
            .merge(dataset, on="userID", suffixes=("", "_0")) \
            .query("timestamp_0 >= timestamp - {} and timestamp_0 < timestamp".format(period_milliseconds)) \
            .drop("timestamp_0", axis=1)

        y_test = x_test[["userID", "timestamp"]].drop_duplicates().merge(dataset, on=["userID", "timestamp"])

        return x_train, x_test, y_train, y_test

    def transform(self, dataset, func=lambda x: np.any(x)):
        transformed_dfs = []
        split_df = self.split(dataset)
        for df in split_df[:-1]:
            df = df.pivot_table(index=["userID", "timestamp"], columns="itemID", values="amount", aggfunc=func)
            abs_hkeys = set(self.levels) - set(df.columns.values)
            for key in abs_hkeys:
                df[key] = False
            transformed_dfs.append(df[self.levels].fillna(False))
        transformed_dfs.append(split_df[-1])
        return transformed_dfs

if __name__ == "__main__":
    transformer = SciPyLearningTransformer()

    dataset = Input("/Users/g.sarapulov/MLProjects/playground/test_transactions.csv").read()
    x_train, x_test, y_train, y_test = transformer.transform(dataset)
    print(x_train.head())
    print(y_test.sort_values(["userID", "timestamp", "itemID"]).head(20))
