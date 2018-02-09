import sklearn.metrics

NUM_ITEMS = len(open("main/items.txt", "r").read().split("\n"))


class Evaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, x, y):
        metrics = {}
        for k in range(1, 11):
            metrics[k] = self.usage_prediction_metrics(x, y, k)
        return metrics

    def usage_prediction_metrics(self, x, y, k):
        left = y.copy()
        left["left"] = 1
        right = self.model.predict(x).query("rank <= {}".format(k))
        right["right"] = 1
        merged = left.merge(right, on=["userID", "timestamp", "itemID"], how="outer")
        merged["tp"] = (merged["left"] == merged["right"]).astype("int")
        merged["fn"] = ((merged["left"] == 1) & (merged["tp"] == 0)).astype("int")
        merged["fp"] = ((merged["right"] == 1) & (merged["tp"] == 0)).astype("int")
        merged["tn"] = NUM_ITEMS - merged["tp"] - merged["fp"] - merged["fn"]

        merged = merged[["userID", "tp", "fp", "tn", "fn"]].groupby("userID").sum()
        merged["precision"] = merged["tp"] / (merged["tp"] + merged["fp"])
        merged["recall"] = merged["tp"] / (merged["tp"] + merged["fn"])
        merged["fpr"] = merged["fp"] / (merged["fp"] + merged["tn"])
        n = merged.shape[0]
        metrics = merged.sum()
        metrics["global_recall"] = metrics["tp"] / (metrics["tp"] + metrics["fn"])
        metrics["global_precision"] = metrics["tp"] / (metrics["tp"] + metrics["fp"])
        metrics["global_fpr"] = metrics["fp"] / (metrics["fp"] + metrics["tn"])
        metrics["precision"] = metrics["precision"] / n
        metrics["recall"] = metrics["recall"] / n
        metrics["fpr"] = metrics["fpr"] / n
        return metrics.to_dict()

    def binary_classification_metrics(self, x, y):
        y_pred = self.model.predict(x) > 0.5
        c = sklearn.metrics.confusion_matrix(y, y_pred)
        tn = int(c[0, 0])
        fp = int(c[0, 1])
        fn = int(c[1, 0])
        tp = int(c[1, 1])

        all_zero = fn + tn == 0
        all_one = fp + tp == 0

        metrics = {
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "precision": tp / (tp + fp) if not all_one else 0,
            "recall": tp / (tp + fn),
            "fpr": fp / (fp + tn),
            "fnr": fn / (fn + tp)
        }
        return metrics
