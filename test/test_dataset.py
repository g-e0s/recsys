import pytest
import pandas as pd
from core.dataset import Dataset

DATA = [
    ('u1', 'i1', 1, 1),
    ('u1', 'i2', 1, 1),
    ('u1', 'i3', 1, 1),
    ('u1', 'i4', 2, 1),

    ('u2', 'i1', 1, 1),
    ('u2', 'i2', 1, 1),
    ('u2', 'i3', 2, 1),
    ('u2', 'i4', 2, 1),
    ('u2', 'i5', 2, 1),

    ('u3', 'i1', 1, 1),
    ('u3', 'i2', 1, 1),
    ('u3', 'i3', 1, 1),
]


def test_xy_split():
    df = pd.DataFrame(DATA, columns=['user', 'item', 'timestamp', 'amount'])
    x, y = Dataset(df).xy_split()
    assert x.users.shape[0] == y.users.shape[0]
    assert x.users.shape[0] == 2
