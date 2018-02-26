import pandas as pd
from sklearn.naive_bayes import BernoulliNB
import implicit

from core.dataset import Dataset
from core.model import SciPyModel
from core.recommender import ClassificationRecommender, FactorizationRecommender
from core.transformer import Pipeline


INPUT_PATH = '/Users/g.sarapulov/MLProjects/playground/sample_transactions.csv'
# read data
train, test = Dataset(pd.read_csv(INPUT_PATH).rename(columns={'userID': 'user', 'itemID': 'item'})).train_test_split()
# define model set
model_set = ClassificationRecommender('nb-set')
for item in train.items.itemID:
    model_set.add(SciPyModel(name='nb_model_{}'.format(item), spec=BernoulliNB()))

# model_set = FactorizationRecommender('als', implicit.als.AlternatingLeastSquares(factors=12))

# build pipeline
pipeline = Pipeline('nb_recommender')
pipeline.add(model_set)

# run pipeline
pipeline.fit(train)
pipeline.validate(test)
