import pytest
from unittest.mock import patch

from main.model import NaiveBayesModel

def test_predict():
    model =