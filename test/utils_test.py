from muttlib import utils
import datetime

def test_str_to_datetime():
	assert utils.str_to_datetime("2019-10-25") == datetime.datetime(2019,10,25)
