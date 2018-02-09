from abc import ABCMeta, abstractmethod, abstractstaticmethod
import numpy as np
import pandas as pd
import keras
import pickle
from sklearn.naive_bayes import BernoulliNB


class Model(metaclass=ABCMeta):
    def __init__(self, spec=None):
        self.model = None
        self._define(spec)

    def _define(self, spec):
        self.model = spec

    @abstractmethod
    def fit(self, x, y, validation_split=None, epochs=None):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def evaluate(self, x, y, metrics):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractstaticmethod
    def load(self, path):
        pass


class SciPyModel(Model):
    def fit(self, x, y, validation_split=None, epochs=None):
        self.model = self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self, x, y, metrics):
        y_pred = self.model.predict(x)
        evaluation_metrics = {}
        for metric in metrics:
            evaluation_metrics[metric.__name__] = metric(y, y_pred)
        return evaluation_metrics

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)


class NaiveBayesModel(SciPyModel):
    def fit(self, x, y, validation_split=None, epochs=None):
        """
        for each category fit NB model and store in dictionary"""
        ml_clf = {}
        for category in y.columns:
            target = y[category].astype(int)
            clf = BernoulliNB().fit(x, target)
            ml_clf[category.replace('_', '')] = clf
        self.model = ml_clf

    def predict(self, x):
        prediction = pd.DataFrame()
        for item, clf in self.model.items():
            df = pd.DataFrame(list(x.index.ravel()), columns=x.index.names)
            df["itemID"] = item
            df["score"] = clf.predict_proba(x)[:, 1]
            prediction = pd.concat([prediction, df.query("score > 0")])
        prediction.reset_index(inplace=True)

        # add rank
        prediction["rank"] = prediction.groupby(["userID", "timestamp"])["score"].rank(method="first", ascending=False)
        print(prediction.head())
        return prediction


class KerasModel(SciPyModel):
    def _define(self, spec):
        if spec["model"] == "sequential":
            _model = keras.models.Sequential()
            for layer in spec["layers"]:
                _model.add(layer)
            _model.compile(optimizer=spec["optimizer"], loss=spec["loss"], metrics=spec["metrics"])
            self.model = _model
        else:
            raise NotImplementedError("`{}` not implemented".format(spec["model"]))

    def fit(self, x, y, validation_split=0.1, epochs=30):
        self.model.fit(x, y, validation_split=validation_split, epochs=epochs)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = keras.models.load_model(path)
