from muttlib import utils
import datetime
import pytest

def test_str_to_datetime():
	assert utils.str_to_datetime("2019-10-25") == datetime.datetime(2019,10,25)
	with pytest.raises(ValueError):
		utils.str_to_datetime("25/10/2019")


