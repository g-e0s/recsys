from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Transformer(metaclass=ABCMeta):
    def __init__(self):
        self.levels = open("items.txt", "r").read().split("\n")


class LearningTransformer(Transformer, metaclass=ABCMeta):
    @abstractmethod
    def split_labels(self, dataset):
        pass

    @abstractmethod
    def train_test_split(self, *df):
        pass

    @abstractmethod
    def transform(self, dataset, func):
        pass


class SciPyLearningTransformer(LearningTransformer):
    def split_labels(self, dataset, period=30):
        period_milliseconds = period * 24 * 60 * 60 * 1000
        orders = dataset[["userID", "timestamp"]].drop_duplicates()
        linked_orders = orders \
            .merge(orders, on="userID", suffixes=("", "_0")) \
            .query("timestamp_0 >= timestamp - {} and timestamp_0 < timestamp".format(period_milliseconds))

        y = linked_orders[["userID", "timestamp"]].merge(dataset, on=["userID", "timestamp"])
        x = linked_orders[["userID", "timestamp_0"]] \
            .rename(columns={"timestamp_0": "timestamp"}) \
            .merge(dataset, on=["userID", "timestamp"])

        return x, y

    def train_test_split(self, *df, ratio=0.8):
        return train_test_split(*df, train_size=ratio)

    def transform(self, dataset, func=lambda x: np.any(x)):
        transformed_dfs = []
        for df in self.split_labels(dataset):
            df = df.pivot_table(index="userID", columns="itemID", values="amount", aggfunc=func)
            abs_hkeys = set(self.levels) - set(df.columns.values)
            for key in abs_hkeys:
                df[key] = False
            transformed_dfs.append(df[self.levels].fillna(False))
        return self.train_test_split(*transformed_dfs)

if __name__ == "__main__":
    transformer = SciPyLearningTransformer()
    dataset = pd.read_csv("/Users/g.sarapulov/MLProjects/playground/test_transactions.csv")
    x_train, x_test, y_train, y_test = transformer.transform(dataset)
    print(x_train.head())
    print(y_train.head())
