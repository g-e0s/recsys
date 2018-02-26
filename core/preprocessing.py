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
        x, y = self.one_hot_encoder.transform(data.xy_split(self.split, self.min_ratings))
        return x, y


class MFPreprocessor(Transformer):
    """transforms rating matrix to sparse scipy matrix"""
    def __init__(self, name='one-hot-encoder', split=0.1, min_ratings=4):
        super().__init__(name)
        self.split = split
        self.min_ratings = min_ratings
        self.ratings = None

    def transform(self, data):
        m = coo_matrix((data.amount, (data.itemID, data.userID)))
        m.data = np.ones(len(m.data))
        m = m.tocsr()
        return m
