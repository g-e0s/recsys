import json
import pickle


class LearningOutput:
    def __init__(self, path):
        self.path = path

    def write_metrics(self, metrics):
        with open(self.path + "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

    def write_model(self, model, model_name, native_save=False):
        if native_save:
            model.save(self.path + "{}.model".format(model_name))
        else:
            with open(self.path + "{}.pickle".format(model_name), "wb") as f:
                pickle.dump(model, f)
