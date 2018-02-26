import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, dataframe, items=None, users=None):
        if items is None:
            self.dataframe, self.users, self.items = self._encode(dataframe)
        else:
            self.dataframe, self.users, self.items = dataframe, users, items
        self.userID = self.dataframe.userID
        self.itemID = self.dataframe.itemID
        self.timestamp = self.dataframe.timestamp
        self.amount = self.dataframe.amount

    def _encode(self, dataframe):
        def _enc(field, idx_name):
            return dataframe[[field]].drop_duplicates() \
                .reset_index().drop('index', axis=1).reset_index().rename(columns={'index': idx_name})
        users = _enc('user', 'userID')
        items = _enc('item', 'itemID')
        encoded = dataframe.merge(users, on='user').merge(items, on='item').drop(['user', 'item'], axis=1)
        return encoded, users, items

    def xy_split(self, split=0.1, min_ratings=4):
        grouped = self.dataframe.groupby('userID')
        enough_ratings = grouped['itemID'].transform(pd.Series.count) >= min_ratings
        grouped = self.dataframe[enough_ratings].groupby('userID')
        x_idx = grouped['timestamp'].rank(method='first', pct=True) < (1 - split)
        x = self.dataframe[enough_ratings][x_idx]
        y = self.dataframe[enough_ratings][~x_idx]
        users = self.users.merge(x[['userID']].drop_duplicates(), on='userID')
        # assert x.shape[0] == y.shape[0]
        return Dataset(x, items=self.items, users=users), Dataset(y, items=self.items, users=users)

    def train_test_split(self, split=0.1):
        random_idx = np.random.binomial(1, split, len(self.users.userID))
        train = self.users[['userID']][random_idx == 0].merge(self.dataframe, on='userID')
        test = self.users[['userID']][random_idx == 1].merge(self.dataframe, on='userID')
        return (Dataset(train, items=self.items, users=self.users[['userID']][random_idx == 0]),
                Dataset(test, items=self.items, users=self.users[['userID']][random_idx == 1]))

    def n_fold_split(self, n=5):
        random_idx = np.random.randint(n-1, size=len(self.users.userID))
        folds = []
        for i in range(n):
            fold = self.users[['userID']][random_idx == i].merge(self.dataframe, on='userID')
            folds.append(Dataset(fold, items=self.items, users=self.users[['userID']][random_idx == i]))
        return folds
