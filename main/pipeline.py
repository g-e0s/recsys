from abc import ABCMeta, abstractmethod
from sklearn.naive_bayes import BernoulliNB

from main.input import Input
from main.evaluation import Evaluator
from main.model import Model, SciPyModel, NaiveBayesModel
from main.output import LearningOutput
from main.transformer import LearningTransformer, SciPyLearningTransformer


class LearningPipeline(metaclass=ABCMeta):
    def __init__(self, input_path, output_path, spec):
        self._set_input_stage(input_path)
        self._set_transform_stage()
        self._set_model_stage(spec)
        self._set_eval_stage()
        self._set_output_stage(output_path)

    @abstractmethod
    def _set_input_stage(self, input_path):
        self.input_stage = None

    @abstractmethod
    def _set_transform_stage(self):
        self.transform_stage = LearningTransformer()

    @abstractmethod
    def _set_model_stage(self, spec):
        self.model_stage = Model(spec)

    @abstractmethod
    def _set_eval_stage(self):
        self.eval_stage = Evaluator()

    @abstractmethod
    def _set_output_stage(self, output_path):
        self.output_stage = None

    @abstractmethod
    def run(self):
        pass


class SciPyLearningPipeline(LearningPipeline):
    def _set_input_stage(self, input_path):
        self.input_stage = Input(input_path)

    def _set_transform_stage(self):
        self.transform_stage = SciPyLearningTransformer()

    def _set_model_stage(self, spec):
        self.model_stage = SciPyModel(spec)

    def _set_eval_stage(self):
        self.eval_stage = Evaluator(self.model_stage)

    def _set_output_stage(self, output_path):
        self.output_stage = LearningOutput(output_path)

    def run(self):
        dataset = self.input_stage.read()
        x_train, x_test, y_train, y_test = self.transform_stage.transform(dataset)
        self.model_stage.fit(x_train, y_train)
        metrics = self.eval_stage.evaluate(x_test, y_test)
        self.output_stage.write(model=self.model_stage, metrics=metrics)


class NaiveBayesLearningPipeline(SciPyLearningPipeline):
    def _set_model_stage(self, spec):
        self.model_stage = NaiveBayesModel(spec)


class PredictingPipeline(metaclass=ABCMeta):
    def __init__(self):
        self.input_stage = None
        self.split_stage = None
        self.transform_stage = None
        self.model_stage = None
        self.eval_stage = None
        self.output_stage = None

    @abstractmethod
    def set_transform_stage(self):
        self.transform_stage = LearningTransformer()

    @abstractmethod
    def set_model_stage(self):
        self.model_stage = Model()

    @abstractmethod
    def set_eval_stage(self):
        self.eval_stage = Evaluator()

    @abstractmethod
    def run(self, input_path, output_path):
        dataset = self.input_stage.read(input_path)
        x_train, x_test, y_train, y_test = self.transform_stage.transform(dataset)
        self.model_stage.fit(x_train, y_train)
        self.eval_stage.evaluate(x_test, y_test)
        self.output_stage.write(output_path)

if __name__ == "__main__":
    pipeline = NaiveBayesLearningPipeline(input_path="/Users/g.sarapulov/MLProjects/playground/test_transactions.csv",
                                          output_path="/Users/g.sarapulov/MLProjects/playground/", spec=BernoulliNB())
    pipeline.run()
