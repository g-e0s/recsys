import pandas as pd
from sklearn.naive_bayes import BernoulliNB

from core.dataset import Dataset
from core.model import SciPyModel, ALSRecommender
from core.recommender import ClassificationRecommender, FactorizationRecommender
from core.transformer import Pipeline


# INPUT_PATH = '/Users/g.sarapulov/MLProjects/playground/sample_transactions.csv'
INPUT_PATH = '/Users/g.sarapulov/MLProjects/playground/transactions.csv'
OUTPUT_PATH = '/Users/g.sarapulov/MLProjects/playground/nb_recs.csv'
# read data
# train, test = Dataset(pd.read_csv(INPUT_PATH, nrows=10000).rename(columns={'userID': 'user', 'itemID': 'item'})).train_test_split()
dataset = Dataset(pd.read_csv(INPUT_PATH))
train, test = dataset.train_test_split()

# define model set
# model_set = ClassificationRecommender('nb-set')
# for item in train.items.itemID:
#     model_set.add(SciPyModel(name='nb_model_{}'.format(item), spec=BernoulliNB()))

model_set = FactorizationRecommender('als', ALSRecommender(factors=8, regularization=0.01, iterations=30))

# build pipeline
pipeline = Pipeline('nb_recommender')
pipeline.add(model_set)

# run pipeline
pipeline.fit(train)
pipeline.validate(test)
# pipeline.predict(dataset).query('rank <= 5') \
#     .merge(dataset.users, on='userID') \
#     .merge(dataset.items, on='itemID')[['user', 'item', 'score', 'rank']] \
#     .to_csv(OUTPUT_PATH, index=False)
