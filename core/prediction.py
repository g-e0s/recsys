import pandas as pd
from core.transformer import SimpleTransformer


class TopItems(SimpleTransformer):
    def transform(self, data):
        prediction = pd.DataFrame()
        for item in data.items:
            df = pd.DataFrame()
            df['userID'] = data.users
            df["itemID"] = item
            df["score"] = data.item
            prediction = pd.concat([prediction, df])
        prediction.reset_index(inplace=True)

        # add rank
        prediction["rank"] = prediction.groupby(["userID", "timestamp"])["score"].rank(method="first", ascending=False)
        print(prediction.head())
        return prediction

