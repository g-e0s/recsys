import pandas as pd
import json
from pyspark.sql import SparkSession, functions as f, types as t

INPUT_PATH = "/Users/g.sarapulov/MLProjects/playground/test_transactions.csv"


class Input:
    def __init__(self, path):
        self.path = path

    def read(self, reader="pandas"):
        if reader == "pandas":
            return pd.read_csv(self.path).groupby(["userID", "timestamp", "itemID"]).sum().reset_index()
        elif reader == "spark":
            spark = SparkSession.builder.getOrCreate()
            schema = t.StructType([
                t.StructField("timestamp", t.LongType()),
                t.StructField("userID", t.StringType()),
                t.StructField("itemID", t.StringType()),
                t.StructField("amount", t.DoubleType()),
                t.StructField("discount", t.DoubleType())
            ])
            df = spark.read.csv("file://" + self.path, schema=schema)
            return df

    def get_iterator(self):
        return self.read(reader="spark") \
            .withColumn("orderDate", (f.col("timestamp") / 1000).cast("timestamp").cast("date")) \
            .groupBy("userID", "itemID", "orderDate") \
            .agg(f.sum(f.col("amount")).alias("amount"), f.sum(f.col("discount")).alias("discount")) \
            .na.drop() \
            .groupBy("userID", "orderDate") \
            .agg(f.collect_list(f.struct("itemID", "amount")).alias("order")) \
            .toLocalIterator()

if __name__ == "__main__":
    from pprint import pprint
    iter = Input(INPUT_PATH).get_iterator()
    pprint(next(iter).asDict(recursive=True))
