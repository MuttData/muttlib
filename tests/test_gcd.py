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
    "ed, twin, fwin, tg, ehour, sd, fd, tn",
    [
        (
            pd.to_datetime('2019-06-01'),
            60,
            7,
            'H',
            0,
            pd.Timestamp("'2019-04-02"),
            pd.to_datetime('2019-06-08'),
            'hourly',
        ),
        (
            pd.to_datetime('2019-06-01'),
            60,
            60,
            'M',
            0,
            pd.Timestamp("'2019-04-02"),
            pd.to_datetime('2019-07-31'),
            'monthly',
        ),
    ],
)
def test_time_range_configuration_init(ed, twin, fwin, tg, ehour, sd, fd, tn):
    trc = TimeRangeConfiguration(ed, twin, fwin, tg, end_hour=ehour)
    assert trc.end_date == ed
    assert trc.start_date == sd
    assert trc.future_date == fd
    assert trc.time_granularity == tg
    assert trc.time_granularity_name == tn


def test_time_range_configuration_init_error():
    with pytest.raises(ValueError):
        TimeRangeConfiguration(pd.to_datetime('2019-06-01'), -2, -1, 'H')
