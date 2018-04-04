import numpy as np
from scipy.sparse import coo_matrix

from core.transformer import Transformer


class OneHot(Transformer):
    def __init__(self, name='one-hot-encoder'):
        super().__init__(name)

    def transform(self, data, func=lambda x: np.any(x)):
        def pivot(df):
            return df.dataframe \
                .pivot_table(index="userID", columns="itemID", values="amount", aggfunc=func) \
                .reindex(df.items.itemID, axis=1).fillna(False).values
        if isinstance(data, (tuple, list)):
            return [pivot(df) for df in data]
        else:
            return pivot(data)


class ClassRSPreprocessor(Transformer):
    def __init__(self, name='one-hot-encoder', split=0.1, min_ratings=4):
        super().__init__(name)
        self.one_hot_encoder = OneHot()
        self.split = split
        self.min_ratings = min_ratings

    def transform(self, data):
        # x, y = self.one_hot_encoder.transform(data.leave_k_out(k=2, time_aware=False))
        x, y = self.one_hot_encoder.transform(data.session_holdout(min_sessions=3))
        return x, y


class MFPreprocessor(Transformer):
    """transforms rating matrix to sparse scipy matrix"""
    def __init__(self, name='one-hot-encoder', split=0.1, min_ratings=4):
        super().__init__(name)
        self.split = split
        self.min_ratings = min_ratings
        self.ratings = None

    def transform(self, data, alpha=40):
        d = data.dataframe.pivot_table(index=['userID', 'itemID'], values=['timestamp', 'amount'],
                                       aggfunc={'timestamp': np.mean, 'amount': np.size}).reset_index()

        m = coo_matrix((1 + alpha * d.amount, (d.itemID, d.userID)))
        # m.data = 1 + alpha * d.amount # np.ones(len(m.data))
        m = m.tocsr()
        return m
