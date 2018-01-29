import os
import pandas as pd
from pyspark.sql import SparkSession, types as t, functions as f


class BatchLayer:
    def __init__(self, spark, storage):
        self.spark = spark
        self.storage = storage

    def factory(self, batch_type):
        if batch_type == "spark":
            return SparkBatchLayer(self.spark, self.storage)
        elif batch_type == "spark":
            return PandasBatchLayer(self.spark, self.storage)
        else:
            raise NotImplementedError("Batch type `{}` not implemented")


class SparkBatchLayer(BatchLayer):
    @staticmethod
    def define_table_schema(storage, table):
        fields = []
        for field, type_string in storage[table]["schema"].items():
            if type_string == "string":
                fields.append(t.StructField(field, t.StringType()))
            elif type_string == "long":
                fields.append(t.StructField(field, t.LongType()))
            elif type_string == "int":
                fields.append(t.StructField(field, t.IntegerType()))
            elif type_string == "double":
                fields.append(t.StructField(field, t.DoubleType()))
            elif type_string == "date":
                fields.append(t.StructField(field, t.DateType()))
            elif type_string == "boolean":
                fields.append(t.StructField(field, t.BooleanType()))
            else:
                raise ValueError("undefined typestring `{}`".format(type_string))
        return t.StructType(fields)

    def read_table(self, table, sep=",", header=True):
        return self.spark.read.csv(path="file://" + os.path.abspath(self.storage[table]["path"]),
                                   sep=sep, header=header, schema=self.define_table_schema(self.storage, table))

    def sample_transaction_data(self, start_ts, end_ts, period_days=30):
        positions = self.read_table("positions")
        items = self.read_table("items")
        orders = positions.join(items, on="productCode") \
            .groupBy("userID", "timestamp") \
            .agg(f.collect_list("itemID").alias("positions"))

        orders.createOrReplaceTempView("orders")

        order_query = """
            SELECT
                o1.userID,
                o1.timestamp,
                o1.positions,
                o2.positions basket
            FROM orders o1
            INNER JOIN orders o2
                ON o1.userID = o2.userID
                AND o2.timestamp BETWEEN o1.timestamp - {0} * 24 * 60 * 60 * 1000 AND o1.timestamp - 1
            WHERE o1.timestamp BETWEEN {0} + {1} * 24 * 60 * 60 * 1000 AND {2}
        """.format(start_ts, period_days, end_ts)

        transactions = self.spark.sql(order_query) \
            .groupBy("userID", "timestamp", "positions") \
            .agg(f.collect_list("basket").alias("basket")) \
            .withColumn("basket", f.udf(lambda nested_list: [x for l in nested_list for x in l])("basket"))

        return transactions


class PandasBatchLayer(BatchLayer):
    @staticmethod
    def define_table_schema(storage, table):
        fields = {}
        for field, type_string in storage[table]["schema"].items():
            if type_string == "string":
                fields[field] = "str"
            elif type_string == "long":
                fields[field] = "int"
            elif type_string == "int":
                fields[field] = "int"
            elif type_string == "double":
                fields[field] = "float"
            elif type_string == "date":
                fields[field] = "datetime64"
            elif type_string == "boolean":
                fields[field] = "str"
            else:
                raise ValueError("undefined typestring `{}`".format(type_string))
        return t.StructType(fields)

    def read_table(self, table, sep, header):
        return pd.read_csv(path=self.storage[table]["path"], sep=sep,
                           header=header, dtype=self.define_table_schema(self.storage, table))

