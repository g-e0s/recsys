import keras.models as m
import pickle

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
