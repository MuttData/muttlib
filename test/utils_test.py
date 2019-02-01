from muttlib import utils
import datetime
import pytest

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


