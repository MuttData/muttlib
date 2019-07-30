"""
Greatest Common Divisor module.

Holds various widely used and tested classes along different projects.
Same with abstract + base classes with different responsibilities.
"""
from muttlib.utils import create_forecaster_dates

from sklearn.base import BaseEstimator


class AttributeHelperMixin(BaseEstimator):
    """Helper mixing to handle params, metrics and artifacts of processing steps.

    This mixing relieves the step of knowning how to log metrics and artifacts
    and allows the possibility of having different backends (instead of just mlflow).
    """

    def _get_init(self, n, v):
        """Initialize instance vars on the fly.

        Not nice but removes the need of cooperation from subclasses.
        """
        if not hasattr(self, n):
            setattr(self, n, v)
        return v

    def _get_init_dict(self, n):
        return self._get_init(n, {})

    def _get_init_list(self, n):
        return self._get_init(n, [])

    def log_artifact(self, path):
        self._get_init_list('_artifacts')
        self._artifacts.append(path)

    def get_artifacts(self):
        self._get_init_list('_artifacts')
        return self._artifacts

    def log_metric(self, k, v):
        self.log_metrics({k: v})

    def log_metrics(self, d):
        self._get_init_dict('_metrics')
        self._metrics.update(d)

    def get_metrics(self):
        self._get_init_dict('_metrics')
        return self._metrics


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
    ):
        """Initialize instance.

            end_date (datetime): End of the data to train.
            forecast_train_window (int): Number of days used for training.
            forecast_future_window (int): Number for the forecast.
            time_granularity (string): String values defining a time frequency
                such as 'H' or 'D'.

        Ref: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        """
        sd, ed, fd = create_forecaster_dates(
            end_date, forecast_train_window, forecast_future_window
        )
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
    def forecast_future_window(self):
        return self._forecast_future_window

    @property
    def forecast_train_window(self):
        return self._forecast_train_window

    @property
    def start_date(self):
        return self._start_date

    @property
    def end_date(self):
        return self._end_date

    @property
    def future_date(self):
        return self._future_date

    @property
    def time_granularity(self):
        return self._time_granularity

    @property
    def time_granularity_name(self):
        return self._time_granularity_name

    def is_hourly(self):
        return self.time_granularity == self.HOURLY_TIME_GRANULARITY

    def is_daily(self):
        return self.time_granularity == self.DAILY_TIME_GRANULARITY

    def is_weekly(self):
        return self.time_granularity == self.WEEKLY_TIME_GRANULARITY

    def is_monthly(self):
        return self.time_granularity == self.MONTHLY_TIME_GRANULARITY


class BaseStrategyBuilder:
    """Abstract class to implement strategies.

    Attributes:
        name (str): Name to be matched by the subclass.
    """

    name = None

    def __init__(self, name):
        self.name = name

    @classmethod
    def accept(cls, name):
        """Check if this class name matches with the one given.

        Returns
            True if `cls.name == name` False otherwise.
        """
        return cls.name == name

    @classmethod
    def soft_accept(cls, name):
        """Check if this class name matches by an alternative criteria.

        Use this to implement partial name matchings for example.

        Returns
            True if class matches the given criteria.
        """
        return cls.accept(name)

    @classmethod
    def build(cls, name):
        """Find matching subclass and instance it.

        This method calls the `accept` method of each subclass to find the one by name.
        If None is found the same is done with `soft_accept`.
        If this also fails the current class will be instantiated.
        """
        scs = cls.__subclasses__()

        scl_hard = [sc for sc in scs if sc.accept(name)]
        scl_soft = [sc for sc in scs if sc.soft_accept(name)]
        for scl in [scl_hard, scl_soft]:
            if len(scl) > 1:
                raise ValueError(
                    f"Multiple subclasses matched {name}. Check your namings."
                )
            if len(scl) == 1:
                sc = scl[0](name)
                break
        else:
            sc = cls

        return sc(name)
