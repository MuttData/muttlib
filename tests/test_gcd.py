import pytest
import pandas as pd

from muttlib.gcd import TimeRangeConfiguration


@pytest.fixture
def trange_conf():
    """Setup code to create a TimeRangeConf object with daily granularity."""
    end_date = pd.to_datetime('2019-06-01')
    granularity = 'D'
    train_window, forecast_window = 60, 7
    trc = TimeRangeConfiguration(end_date, train_window, forecast_window, granularity)
    return trc


def test_no_value():
    with pytest.raises(Exception):
        TimeRangeConfiguration()  # pylint: disable=no-value-for-parameter


@pytest.mark.parametrize(
    "sd, ed, fd",
    [
        (
            pd.Timestamp("'2019-04-02"),
            pd.to_datetime('2019-06-01'),
            pd.to_datetime('2019-06-08'),
        )
    ],
)
def test_correct_dates(trange_conf, sd, ed, fd):  # pylint: disable=redefined-outer-name
    assert trange_conf.start_date == sd
    assert trange_conf.end_date == ed
    assert trange_conf.future_date == fd
