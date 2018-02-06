import pickle


class LearningOutput:
    def __init__(self, path):
        self.path = path

    def write(self, model, metrics):
        with open(self.path + "model.pickle", "wb") as f:
            pickle.dump(model, f)
        with open(self.path + "metrics.txt", "w") as f:
            f.write(metrics)
