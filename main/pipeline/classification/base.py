from main.pipeline.base import SciPyLearningPipeline
from main.model.model import NaiveBayesModel


class ClassificationPipeline(SciPyLearningPipeline):
    def _set_model_stage(self, spec):
        self.model_stage = NaiveBayesModel(spec)
