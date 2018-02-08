import json
import pickle


class LearningOutput:
    def __init__(self, path):
        self.path = path

    def write(self, model, metrics):
        with open(self.path + "model.pickle", "wb") as f:
            pickle.dump(model, f)
        with open(self.path + "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
