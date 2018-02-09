from sklearn.naive_bayes import BernoulliNB

from main.pipeline.base import NaiveBayesLearningPipeline

INPUT_PATH = "/Users/g.sarapulov/MLProjects/playground/test_transactions.csv"
OUTPUT_PATH = "/Users/g.sarapulov/MLProjects/playground/"

pipeline = NaiveBayesLearningPipeline(input_path=INPUT_PATH, output_path=OUTPUT_PATH, spec=BernoulliNB())
pipeline.run()
