import pyspark.sql as ssql
import pandas as pd
import numpy as np


class SparkInput:
    def __init__(self):
        self.spark = ssql.SparkSession.builder.getOrCreate()

    def read_csv(self, path, schema, **kwargs):
        return self.spark.read.csv(path, schema=schema, **kwargs)

    def get_rows_iterator(self, df):
        return df.toLocalIterator()




