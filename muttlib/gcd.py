"""Greatest Common Divisor module.

Holds various widely used and tested classes along different projects.
Same with abstract + base classes with different responsibilities.


Example using timeRangeConf:

> from muttlib.gcd import TimeRangeConfiguration
> import pandas as pd
> end_date = pd.to_datetime('2019-07-01')
> trc = TimeRangeConfiguration(end_date, 365*4, 365, 'M')
>
> print(trc.start_date, trc.end_date, trc.future_date, trc.is_monthly())
2015-07-02 00:00:00 2019-07-01 00:00:00 2020-06-30 00:00:00 True

DeprecationWarning:

This module will be removed from muttlib in version 1.0.0
"""
import warnings

warnings.warn(
    "This module will be removed from muttlib in version 1.0.0",
    DeprecationWarning,
    stacklevel=2,
)
from muttlib.utils import create_forecaster_dates
from sklearn.base import BaseEstimator
from deprecated import deprecated


@deprecated(
    reason="'AttributeHelperMixin' will be removed from muttlib in version 1.0.0"
)
class AttributeHelperMixin(BaseEstimator):
    """Helper mixing to handle params, metrics and artifacts from processing steps.

    This mixing relieves the step of knowning how to log metrics and artifacts
    and allows the possibility of having different backends (instead of just mlflow).
    """

    def _get_init(self, name, val):
        """Initialize instance vars on the fly.

        Args:
            name (str): Attribute's name.
            val (dict, list): Desired attr value to set.

        Notes:
            Not nice but removes the need of cooperation from subclasses.

        """
        if not hasattr(self, name):
            setattr(self, name, val)
        return val

    def _get_init_dict(self, name):  # noqa: D102
        return self._get_init(name, val={})

    def _get_init_list(self, name):  # noqa: D102
        return self._get_init(name, val=[])

    def log_artifact(self, path):  # noqa: D102
        self._get_init_list('_artifacts')
        self._artifacts.append(path)

    def get_artifacts(self):  # noqa: D102
        self._get_init_list('_artifacts')
        return self._artifacts

    def log_metrics(self, d):  # noqa: D102
        self._get_init_dict('_metrics')
        self._metrics.update(d)

    def log_metric(self, k, v):  # noqa: D102
        self.log_metrics({k: v})

    def get_metrics(self):  # noqa: D102
        self._get_init_dict('_metrics')
        return self._metrics


@deprecated(
    reason="'TimeRangeConfiguration' will be removed from muttlib in version 1.0.0"
)
class TimeRangeConfiguration(AttributeHelperMixin):
    """Time configurations that should remain constant when reprocessing."""

    HOURLY_TIME_GRANULARITY = 'H'
    DAILY_TIME_GRANULARITY = 'D'
    WEEKLY_TIME_GRANULARITY = 'W'
    MONTHLY_TIME_GRANULARITY = 'M'

    TIME_GRANULARITY_NAME_MAP = {
        HOURLY_TIME_GRANULARITY: 'hourly',
        DAILY_TIME_GRANULARITY: 'daily',
        WEEKLY_TIME_GRANULARITY: 'weekly',
        MONTHLY_TIME_GRANULARITY: 'monthly',
    }
    TIME_GRANULARITIES = list(TIME_GRANULARITY_NAME_MAP.keys())

    def __init__(
        self,
        end_date,
        forecast_train_window,
        forecast_future_window,
        time_granularity,
        end_hour=0,
        round_dates=False,
    ):
        """Initialize.

        Parameters
        ----------
        end_date : datetime
            End of the data to train.
        forecast_train_window : int
            Number of days previous to the `end_date`, used for training.
        forecast_future_window : int
            Positive number of future days for the forecast interval.
            Note this is anchored from the `end_date` arg.
        time_granularity : string:
            String values defining a time frequency such as 'H', 'D', 'M', etc.

        Notes
        -----
        Resulting calculated dates will be rounded to the nearest **lowest** time,
        according to the granularity.

        Refs
        * https://pandas.pydata.org/pandas-docs/stable/user_guide/
        * timeseries.html#timeseries-offset-aliases
        """
        sd, ed, fd = create_forecaster_dates(
            end_date, forecast_train_window, forecast_future_window
        )
        if round_dates is True:
            sd, fd = [d.to_period(time_granularity).to_timestamp() for d in (sd, fd)]

        self._start_date, self._end_date, self._future_date = (
            sd,
            ed.replace(hour=end_hour),
            fd,
        )
        self._forecast_train_window = forecast_train_window
        self._forecast_future_window = forecast_future_window
        self._time_granularity = time_granularity
        self._time_granularity_name = self.TIME_GRANULARITY_NAME_MAP[time_granularity]
        self._validate_construction()

    def _validate_construction(self):
        """Validate current configuration."""
        if not self._start_date <= self._end_date:
            raise ValueError(  # type: ignore
                f'Bad dates passed. Start:{self._start_date}, end:{self._end_date}.'
            )
        if not self._end_date <= self._future_date:
            raise ValueError(  # type: ignore
                f'Bad dates passed. End:{self._end_date}, future:{self._future_date}.'
            )
        if self._time_granularity not in self.TIME_GRANULARITIES:
            raise ValueError(  # type: ignore
                f'Bad time granularity passed:{self._time_granularity}, '
                f'Possible values are: {self.TIME_GRANULARITIES}\n'
            )

    @property
    def forecast_future_window(self):  # noqa: D102
        return self._forecast_future_window

    @property
    def forecast_train_window(self):  # noqa: D102
        return self._forecast_train_window

    @property
    def start_date(self):  # noqa: D102
        return self._start_date

    @property
    def end_date(self):  # noqa: D102
        return self._end_date

    @property
    def future_date(self):  # noqa: D102
        return self._future_date

    @property
    def time_granularity(self):  # noqa: D102
        return self._time_granularity

    @property
    def time_granularity_name(self):  # noqa: D102
        return self._time_granularity_name

    def is_hourly(self):
        """Check if granularity is of hourly type."""
        return self.time_granularity == self.HOURLY_TIME_GRANULARITY

    def is_daily(self):
        """Check if granularity is of daily type."""
        return self.time_granularity == self.DAILY_TIME_GRANULARITY

    def is_weekly(self):
        """Check if granularity is of weekly type."""
        return self.time_granularity == self.WEEKLY_TIME_GRANULARITY

    def is_monthly(self):
        """Check if granularity is of monthly type."""
        return self.time_granularity == self.MONTHLY_TIME_GRANULARITY
