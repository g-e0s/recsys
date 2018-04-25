import sklearn.metrics

# NUM_ITEMS = len(open("main/items.txt", "r").read().split("\n"))


class Evaluator:
    def evaluate(self, true, pred):
        metrics = {}
        for k in range(1, 11):
            metrics[k] = self.usage_prediction_metrics(true, pred, k)
        return metrics

    def usage_prediction_metrics(self, true, pred, k):
        num_items = len(set(list(true.itemID) + list(pred.itemID)))
        left = true.dataframe.copy()
        left["left"] = 1
        right = pred.copy()
        # add rank
        right["rank"] = pred.groupby("userID")["score"].rank(method="first", ascending=False).astype(int)
        right = right.query("rank <= {}".format(k))
        right["right"] = 1
        merged = left.merge(right, on=["userID", "itemID"], how="outer")
        merged["tp"] = (merged["left"] == merged["right"]).astype("int")
        merged["fn"] = ((merged["left"] == 1) & (merged["tp"] == 0)).astype("int")
        merged["fp"] = ((merged["right"] == 1) & (merged["tp"] == 0)).astype("int")
        merged["tn"] = num_items - merged["tp"] - merged["fp"] - merged["fn"]

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
