# import keras.models as m
import pickle
import numpy as np
import itertools

from implicit.als import AlternatingLeastSquares

from core.transformer import Transformer


class Model(Transformer):
    _transformer_type = 'model'

    def __init__(self, name, spec=None):
        super().__init__(name)
        self.spec = spec
        self.model = None
        self.items = None

    def load_model(self, path):
        pass

    def save_model(self, path):
        pass

    def fit(self, x, y=None, **kwargs):
        pass

    def predict(self, x):
        pass

    def transform(self, x):
        pass


class KerasModel(Model):
    def _define(self, spec):
        if spec["model"] == "sequential":
            _model = m.Sequential()
            for layer in spec["layers"]:
                _model.add(layer)
            _model.compile(optimizer=spec["optimizer"], loss=spec["loss"], metrics=spec["metrics"])
            self.model = _model
        else:
            raise NotImplementedError("`{}` not implemented".format(spec["model"]))

    def fit(self, x, y=None, **kwargs):
        if not self.model:
            self.model = self._define(self.spec)
        self.model.fit(x, y, **kwargs)

    def load(self, path):
        self.model = m.load_model(path)

    def save(self, path):
        self.model.save(path)


class SciPyModel(Model):
    def fit(self, x, y=None, **kwargs):
        if not self.model:
            self.model = self.spec
        self.model.fit(x, y, **kwargs)

    def predict(self, x):
        return self.model.predict_proba(x)[:, 1]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)


class ALSRecommender(AlternatingLeastSquares):
    """class inherited from implicit's ALS with added possibility to leave previously liked items"""
    def recommend(self, userid, user_items, N=10, filter_items=None, recalculate_user=False, filter_liked=False):
        def is_available(rec):
            return rec not in liked if filter_liked else True
        user = self._user_factor(userid, user_items, recalculate_user)

        # calculate the top N items, removing the users own liked items from the results
        liked = set(user_items[userid].indices)
        scores = self.item_factors.dot(user)
        if filter_items:
            liked.update(filter_items)

        count = N + len(liked)
        if count < len(scores):
            ids = np.argpartition(scores, -count)[-count:]
            best = sorted(zip(ids, scores[ids]), key=lambda x: -x[1])
        else:
            best = sorted(enumerate(scores), key=lambda x: -x[1])
        return list(itertools.islice((rec for rec in best if is_available(rec[0])), N))
