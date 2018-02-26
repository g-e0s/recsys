import pandas as pd

from core.transformer import Transformer, ParallelTransformer
from core.preprocessing import ClassRSPreprocessor, MFPreprocessor
from core.evaluation import Evaluator
from core.model import Model


class Recommender(Model):

    def __init__(self, name, spec=None):
        super().__init__(name, spec)
        self.items = None
        self.evaluator = Evaluator()

    def recommend_for_user(self, user_id, n):
        raise NotImplementedError

    def recommend_for_item(self, item_id, n):
        raise NotImplementedError

    def validate(self, data):
        x, y = data.xy_split()
        prediction = self.predict(x)
        metrics = self.evaluator.evaluate(y, prediction)
        print(pd.DataFrame.from_dict(metrics, orient='index').drop(['tp', 'fp', 'tn', 'fn'], axis=1))
        return metrics


# Model-Based recommenders
class ClassificationRecommender(ParallelTransformer, Recommender):
    """Class for classification-based recommender"""
    _recommender_type = 'model_based'

    def __init__(self, name):
        super().__init__(name)
        self.preprocessor = ClassRSPreprocessor()

    def fit(self, x, y=None, **kwargs):
        self.items = x.items
        x, y = self.preprocessor.transform(x)
        for i, model in enumerate(self.transformers):
            model.fit(x, y[:, i])
        return self

    def predict(self, data):
        preprocessed = self.preprocessor.one_hot_encoder.transform(data)
        prediction = pd.DataFrame()

        for i, item in enumerate(self.items.itemID):
            model = self.transformers[i]
            df = pd.DataFrame()
            df['userID'] = data.users.userID
            df["itemID"] = item
            df["score"] = model.predict(preprocessed)
            prediction = pd.concat([prediction, df])
        prediction.reset_index(inplace=True)

        # add rank
        prediction["rank"] = prediction.groupby("userID")["score"].rank(method="first", ascending=False).astype(int)
        return prediction.drop('index', axis=1)


# Matrix factorization recommenders
class FactorizationRecommender(Recommender):
    _recommender_type = 'matrix_factorizer'

    def __init__(self, name, spec):
        super().__init__(name, spec)
        self.model = spec
        self.preprocessor = MFPreprocessor()

    def fit(self, x, y=None, **kwargs):
        x = self.preprocessor.transform(x)
        self.model.fit(x)

    def predict(self, x):
        m = self.preprocessor.transform(x)
        recs = []
        for userid in x.users['userID']:
            user_recs = self.model.recommend(userid, m.T, N=x.items.itemID.size)
            recs += [(userid, *rec) for rec in user_recs]
        df = pd.DataFrame(recs, columns=['userID', 'itemID', 'score'])
        df['rank'] = df.groupby('userID')['score'].rank(ascending=False, method='first').astype(int)
        return df
