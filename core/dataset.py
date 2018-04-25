import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, dataframe, items=None, users=None,
                 user_col='user', item_col='item', rating_col='amount', ts_col='timestamp'):
        renaming_map = {
            user_col: 'user',item_col:'item',rating_col:'amount',ts_col:'timestamp'
        }
        if items is None:
            self.dataframe, self.users, self.items = self._encode(dataframe.rename(columns=renaming_map))
        else:
            self.dataframe, self.users, self.items = dataframe, users, items
        self.userID = self.dataframe.userID
        self.itemID = self.dataframe.itemID
        self.timestamp = self.dataframe.timestamp
        self.amount = self.dataframe.amount

    def _encode(self, dataframe, item_col='item', user_col='user'):
        def _enc(field, idx_name):
            # validate field
            if not isinstance(field, (str, list, tuple, set)):
                raise ValueError('unsupported field type: {}'.format(type(field)))
            col_idx = [field] if isinstance(field, str) else list(field)
            return dataframe[col_idx].drop_duplicates() \
                .reset_index().drop('index', axis=1).reset_index().rename(columns={'index': idx_name})

        try:
            users = _enc(user_col, 'userID')
            items = _enc(item_col, 'itemID')
            dropped_columns = [user_col, item_col] if isinstance(item_col, str) else [user_col] + item_col
            encoded = dataframe.merge(users, on=user_col).merge(items, on=item_col).drop(dropped_columns, axis=1)
            return encoded, users, items
        except KeyError as err:
            print(dataframe.columns, err)

    def leave_k_out(self, k=1, time_aware=True, min_hist=3):
        enough_ratings = self.dataframe.groupby('userID')['itemID'].transform(pd.Series.count) >= min_hist + k
        filtered = self.dataframe[enough_ratings]
        users = self.users.merge(filtered[['userID']].drop_duplicates(), on='userID')
        if time_aware:
            filtered['rank'] = filtered.groupby('userID')['timestamp'].rank(method='first')
        else:
            filtered['rnd'] = np.random.rand(filtered.userID.size)
            filtered['rank'] = filtered.groupby('userID')['timestamp'].rank(method='first')

        filtered['max_rank'] = filtered.groupby('userID')['rank'].transform(pd.Series.max)
        idx = filtered['rank'] + k <= filtered['max_rank']
        x = filtered[idx][self.dataframe.columns]
        y = filtered[~idx][self.dataframe.columns]
        return Dataset(x, items=self.items, users=users), Dataset(y, items=self.items, users=users)

    def session_holdout(self, min_sessions=2):
        df = self.dataframe.copy()
        df['rank'] = df.groupby('userID')['timestamp'].rank(method='dense')
        df['max_rank'] = df.groupby('userID')['rank'].transform(pd.Series.max)
        df = df.query('max_rank > {}'.format(min_sessions))
        users = self.users.merge(df[['userID']].drop_duplicates(), on='userID')
        idx = df['rank'] < df['max_rank']
        return Dataset(df[idx].groupby(['userID', 'itemID']).sum().reset_index(), items=self.items, users=users),\
               Dataset(df[~idx].groupby(['userID', 'itemID']).sum().reset_index(), items=self.items, users=users)

    def train_test_split(self, split=0.1):
        return self.n_fold_split(n=2, weights=[1 - split, split])

    def n_fold_split(self, n=2, weights=None):
        random_idx = np.random.choice(np.arange(n), size=len(self.users.userID), replace=True, p=weights)
        folds = []
        for i in range(n):
            users = self.users[random_idx == i]
            fold = users[['userID']].merge(self.dataframe, on='userID')
            folds.append(Dataset(fold, items=self.items, users=users))
        return folds
