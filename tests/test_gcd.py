import pytest
import pandas as pd

from muttlib.gcd import TimeRangeConfiguration


@pytest.fixture
def daily_trc():
    """Do code setup to create a TimeRangeConf object with daily granularity."""
    end_date = pd.to_datetime('2019-06-01')
    granularity = 'D'
    train_window, forecast_window = 60, 7
    trc = TimeRangeConfiguration(end_date, train_window, forecast_window, granularity)
    return trc


@pytest.fixture
def weekly_trc():
    """Do code setup code to create a TimeRangeConf object with weekly granularity."""
    end_date = pd.to_datetime('2019-06-01')
    granularity = 'W'
    train_window, forecast_window = 120, 21
    trc = TimeRangeConfiguration(end_date, train_window, forecast_window, granularity)
    return trc


def test_no_value():
    with pytest.raises(Exception):
        TimeRangeConfiguration()  # pylint: disable=no-value-for-parameter


def assert_not_all_except_true_one_attr(obj, attr_l, this_attr):
    assert getattr(obj, this_attr)()
    for a in [a for a in attr_l if a != this_attr]:
        assert not getattr(obj, a)()


@pytest.mark.parametrize(
    "ed, twin, fwin, tg, ehour, ro_date, sd, fd, tn",
    [
        (
            pd.to_datetime('2019-06-01'),
            60,
            7,
            'H',
            0,
            True,
            pd.Timestamp("2019-04-02"),
            pd.Timestamp('2019-06-08'),
            'hourly',
        ),
        (
            pd.to_datetime('2019-06-01'),
            60,
            60,
            'M',
            0,
            False,
            pd.Timestamp("2019-04-02"),
            pd.Timestamp('2019-07-31'),
            'monthly',
        ),
        (
            pd.to_datetime('2019-06-01'),
            35,
            21,
            'W',
            0,
            True,
            pd.Timestamp("2019-04-22"),
            pd.Timestamp('2019-06-17'),
            'weekly',
        ),
    ],
)
def test_time_range_configuration_init(ed, twin, fwin, tg, ehour, ro_date, sd, fd, tn):
    trc = TimeRangeConfiguration(ed, twin, fwin, tg, ehour, ro_date)
    assert trc.end_date == ed
    assert trc.start_date == sd
    assert trc.future_date == fd
    assert trc.time_granularity == tg
    assert trc.time_granularity_name == tn
    time_gran_methods = ["is_hourly", "is_daily", "is_weekly", "is_monthly"]
    if tg == 'H':
        assert_not_all_except_true_one_attr(trc, time_gran_methods, 'is_hourly')
    elif tg == 'D':
        assert_not_all_except_true_one_attr(trc, time_gran_methods, 'is_daily')
    elif tg == 'W':
        assert_not_all_except_true_one_attr(trc, time_gran_methods, 'is_weekly')
    elif tg == 'M':
        assert_not_all_except_true_one_attr(trc, time_gran_methods, 'is_monthly')


@pytest.mark.parametrize(
    "ed, twin, fwin, tg, expected_error",
    [
        (pd.to_datetime('2019-06-01'), 2, -1, 'H', ValueError),  # bad windows
        (pd.to_datetime('2019-06-01'), -2, -1, 'H', ValueError),  # bad windows
        (pd.to_datetime('2019-06-01'), -20, 1, 'H', ValueError),  # bad windows
        (pd.to_datetime('2019-06-01'), 2, 1, 'J', KeyError),  # bad granularity
        (pd.to_datetime('2019-06-01'), 2, 1, 'foo', KeyError),  # bad granularity
        (pd.to_datetime('2019-06-01'), 2, 1, 4, KeyError),  # bad granularity
        (0, 2, 1, 4, TypeError),  # bad end-date
        ("foo", 2, 1, 4, ValueError),  # bad end-date
    ],
)
def test_time_range_configuration_init_error(ed, twin, fwin, tg, expected_error):
    with pytest.raises(expected_error):
        TimeRangeConfiguration(ed, twin, fwin, tg)
