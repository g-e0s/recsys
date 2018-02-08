import pandas as pd


class Input:
    def __init__(self, path):
        self.path = path

    def read(self):
        return pd.read_csv(self.path).groupby(["userID", "timestamp", "itemID"]).sum().reset_index()
