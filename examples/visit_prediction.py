import keras.layers as L

from main.data_access_layer.input import Input
from main.data_access_layer.output import LearningOutput
from main.model.model import KerasModel
from main.transform.item_user_image import ItemUserImage
from main.pipeline.base import LearningPipeline
from main.evaluate.evaluation import Evaluator


MODEL_SPEC = {
    "model": "sequential",
    "layers": [
        L.InputLayer(input_shape=(28, 30, 1)),
        L.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        L.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        L.MaxPool2D(pool_size=(2, 2)),
        L.Flatten(),
        L.Dropout(0.4),
        L.Dense(256, activation='relu'),
        L.Dropout(0.3),
        L.Dense(64, activation='relu'),
        L.Dropout(0.2),
        L.Dense(1, activation="sigmoid")
    ],
    "loss": "binary_crossentropy",
    "optimizer": "adam",
    "metrics": ["accuracy"]
}
INPUT_PATH = "/Users/g.sarapulov/MLProjects/playground/test_transactions.csv"
OUTPUT_PATH = "/Users/g.sarapulov/MLProjects/playground/vp/"


class VisitPredictionPipeline(LearningPipeline):
    def _set_input_stage(self, input_path):
        self.input_stage = Input(input_path)

    def _set_transform_stage(self):
        self.transform_stage = ItemUserImage()

    def _set_model_stage(self, spec):
        self.model_stage = {
            "R": KerasModel(spec),
            "I": KerasModel(spec)
        }

    def _set_eval_stage(self):
        self.eval_stage = {"R": Evaluator(self.model_stage["R"]), "I": Evaluator(self.model_stage["I"])}

    def _set_output_stage(self, output_path):
        self.output_stage = LearningOutput(output_path)

    def run(self):
        iterator = Input(INPUT_PATH).get_iterator()
        dataset = [row.asDict(recursive=True) for row in iterator]
        reg, irreg = ItemUserImage().transform(dataset)
        x_train, x_test, y_train, y_test = reg
        self.model_stage["R"].fit(x_train, y_train, epochs=1)
        metrics = dict(R=None, I=None)
        metrics["R"] = self.eval_stage["R"].binary_classification_metrics(x_test, y_test)
        self.output_stage.write_model(self.model_stage["R"], "regular", native_save=True)

        x_train, x_test, y_train, y_test = irreg
        self.model_stage["I"].fit(x_train, y_train, epochs=1)
        metrics["I"] = self.eval_stage["I"].binary_classification_metrics(x_test, y_test)
        self.output_stage.write_model(self.model_stage["I"], "irregular", native_save=True)

        self.output_stage.write_metrics(metrics)

if __name__ == "__main__":
    vp = VisitPredictionPipeline(input_path=INPUT_PATH, output_path=OUTPUT_PATH, spec=MODEL_SPEC)
    vp.run()
