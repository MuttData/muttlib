from muttlib import utils
import datetime
import pytest
from collections import namedtuple
import pandas as pd
import numpy as np

def test_str_to_datetime():
    assert utils.str_to_datetime('2019-10-25 18:35:22') == datetime.datetime(2019, 10, 25, 18, 35, 22)
    assert utils.str_to_datetime('2019-10-25') == datetime.datetime(2019, 10, 25, 0, 0)
    assert utils.str_to_datetime('2019-10-25 18:35:22.000333') == datetime.datetime(2019, 10, 25, 18, 35, 22, 333)
    assert utils.str_to_datetime('18:35:22.000333') == datetime.datetime(1900, 1, 1, 18, 35, 22, 333)
    assert utils.str_to_datetime('18:35:22') == datetime.datetime(1900, 1, 1, 18, 35, 22)
    assert utils.str_to_datetime('20191025T18:35:22') == datetime.datetime(2019, 10, 25, 18, 35, 22)
    assert utils.str_to_datetime('2019-10-25T18:35:22') == datetime.datetime(2019, 10, 25, 18, 35, 22)
    assert utils.str_to_datetime('20191025') == datetime.datetime(2019, 10, 25, 0, 0)
    assert utils.str_to_datetime('2019-10-25T18') == datetime.datetime(2019, 10, 25, 18, 0)
    assert utils.str_to_datetime('201910') == datetime.datetime(2019, 10, 1, 0, 0)
    with pytest.raises(ValueError):
        utils.str_to_datetime("25/10/2019")

def test_dict_to_namedtuple():
    tuple_1 = namedtuple('pepe','don_jose') ('paso por mi casa') 
    tuple_2 = utils.dict_to_namedtuple('pepe',{'don_jose': 'paso por mi casa'})
    assert tuple_1 == tuple_2

def test_create_dict_id():
    assert '67fe8ab7b5' == utils.create_dict_id({'don_jose': 'paso por mi casa'})

def test_make_dirs():
    # Idk :(
    pass

def test_get_ordered_factor_levels():
    data = {'B': [25, 94, 57, 62, 70]}
    df = pd.DataFrame(data, columns = ['B'])
    df2 = utils.get_ordered_factor_levels(df,'B')
    assert np.array_equal(np.array([70, 94, 62, 25, 57]),df2[0])
    assert 5 == df2[1] 
