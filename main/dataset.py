from abc import ABCMeta, abstractmethod, abstractstaticmethod
import pandas as pd
import numpy as np


class DataSet(metaclass=ABCMeta):
    def __init__(self, data=None):
        self.data = data

    @abstractstaticmethod
    def read_data(self, reader, path):
        pass

    @abstractmethod
    def save_data(self, writer, path):
        pass


class PandasDataSet(DataSet):
    def read_data(self, reader, path):
        self.data = pd.read_csv(path, sep=reader.sep)

    def save_data(self, writer, path):
        if writer.file_format == "csv":
            self.data.to_csv(path)
        else:
            raise NotImplementedError


class Reader:
    def __init__(self, file_format="csv", sep=",", decimal="."):
        self.file_format = file_format
        self.sep = sep
        self.decimal = decimal
