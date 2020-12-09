"""muttlib.plotting test suite.

`muttlib` uses `pytest-mpl` to plots testing.

To use, you simply need to mark the function where you want to compare images using
@pytest.mark.mpl_image_compare, and make sure that the function returns
a Matplotlib figure (or any figure object that has a savefig method):

```python
import pytest
import matplotlib.pyplot as plt

@pytest.mark.mpl_image_compare
def test_succeeds():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot([1,2,3])
    return fig
```

To generate the baseline images, run the tests with the --mpl-generate-path option
with the name of the directory where the generated images should be placed:

```python
pytest --mpl-generate-path=baseline
```

More info about `pytest-mpl` library: https://github.com/matplotlib/pytest-mpl#using
"""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from muttlib.plotting import plot
from muttlib.plotting.constants import (
    DAILY_TIME_GRANULARITY,
    HOURLY_TIME_GRANULARITY,
    PLOT_CONFIG,
    DS_COL,
    Y_COL,
    YHAT_COL,
)


@pytest.fixture
def sample_data_df():
    # Taken from https://raw.githubusercontent.com/facebook/prophet/master/examples/example_retail_sales.csv
    return pd.DataFrame.from_records(
        np.array(
            [
                ('2013-02-01T00:00:00.000000000', 373938),
                ('2013-03-01T00:00:00.000000000', 421638),
                ('2013-04-01T00:00:00.000000000', 408381),
                ('2013-05-01T00:00:00.000000000', 436985),
                ('2013-06-01T00:00:00.000000000', 414701),
                ('2013-07-01T00:00:00.000000000', 422357),
                ('2013-08-01T00:00:00.000000000', 434950),
                ('2013-09-01T00:00:00.000000000', 396199),
                ('2013-10-01T00:00:00.000000000', 415740),
                ('2013-11-01T00:00:00.000000000', 423611),
                ('2013-12-01T00:00:00.000000000', 477205),
                ('2014-01-01T00:00:00.000000000', 383399),
                ('2014-02-01T00:00:00.000000000', 380315),
                ('2014-03-01T00:00:00.000000000', 432806),
                ('2014-04-01T00:00:00.000000000', 431415),
                ('2014-05-01T00:00:00.000000000', 458822),
                ('2014-06-01T00:00:00.000000000', 433152),
                ('2014-07-01T00:00:00.000000000', 443005),
                ('2014-08-01T00:00:00.000000000', 450913),
                ('2014-09-01T00:00:00.000000000', 420871),
                ('2014-10-01T00:00:00.000000000', 437702),
                ('2014-11-01T00:00:00.000000000', 437910),
                ('2014-12-01T00:00:00.000000000', 501232),
                ('2015-01-01T00:00:00.000000000', 397252),
                ('2015-02-01T00:00:00.000000000', 386935),
                ('2015-03-01T00:00:00.000000000', 444110),
                ('2015-04-01T00:00:00.000000000', 438217),
                ('2015-05-01T00:00:00.000000000', 462615),
                ('2015-06-01T00:00:00.000000000', 448229),
                ('2015-07-01T00:00:00.000000000', 457710),
                ('2015-08-01T00:00:00.000000000', 456340),
                ('2015-09-01T00:00:00.000000000', 430917),
                ('2015-10-01T00:00:00.000000000', 444959),
                ('2015-11-01T00:00:00.000000000', 444507),
                ('2015-12-01T00:00:00.000000000', 518253),
                ('2016-01-01T00:00:00.000000000', 400928),
                ('2016-02-01T00:00:00.000000000', 413554),
                ('2016-03-01T00:00:00.000000000', 460093),
                ('2016-04-01T00:00:00.000000000', 450935),
                ('2016-05-01T00:00:00.000000000', 471421),
            ],
            dtype=[('ds', '<M8[ns]'), ('y', '<i8')],
        ),
    )


@pytest.fixture
def sample_data_yhat_df():
    # Taken from `sample_data_df`
    return pd.DataFrame.from_records(
        np.array(
            [
                ('2013-02-01T00:00:00.000000000', 3.7394, 0.3739, 7.4788, 1),
                ('2013-03-01T00:00:00.000000000', 4.2164, 0.4216, 8.4328, 1),
                ('2013-04-01T00:00:00.000000000', 4.0838, 0.4084, 8.1676, 1),
                ('2013-05-01T00:00:00.000000000', 4.3699, 0.4370, 8.7397, 1),
                ('2013-06-01T00:00:00.000000000', 4.1470, 0.4147, 8.2940, 1),
                ('2013-07-01T00:00:00.000000000', 4.2236, 0.4224, 8.4471, 1),
                ('2013-08-01T00:00:00.000000000', 4.3495, 0.4350, 8.6990, 1),
                ('2013-09-01T00:00:00.000000000', 3.9620, 0.3962, 7.9240, 1),
                ('2013-10-01T00:00:00.000000000', 4.1574, 0.4157, 8.3148, 1),
                ('2013-11-01T00:00:00.000000000', 4.2361, 0.4236, 8.4722, 1),
                ('2013-12-01T00:00:00.000000000', 4.7721, 0.4772, 9.5441, 1),
                ('2014-01-01T00:00:00.000000000', 3.8340, 0.3834, 7.6680, 1),
                ('2014-02-01T00:00:00.000000000', 3.8032, 0.3803, 7.6063, 1),
                ('2014-03-01T00:00:00.000000000', 4.3281, 0.4328, 8.6561, 1),
                ('2014-04-01T00:00:00.000000000', 4.3142, 0.4314, 8.6283, 1),
                ('2014-05-01T00:00:00.000000000', 4.5882, 0.4588, 9.1764, 1),
                ('2014-06-01T00:00:00.000000000', 4.3315, 0.4332, 8.6630, 1),
                ('2014-07-01T00:00:00.000000000', 4.4301, 0.4430, 8.8601, 1),
                ('2014-08-01T00:00:00.000000000', 4.5091, 0.4509, 9.0183, 1),
                ('2014-09-01T00:00:00.000000000', 4.2087, 0.4209, 8.4174, 1),
                ('2014-10-01T00:00:00.000000000', 4.3770, 0.4377, 8.7540, 1),
                ('2014-11-01T00:00:00.000000000', 4.3791, 0.4379, 8.7582, 1),
                ('2014-12-01T00:00:00.000000000', 5.0123, 0.5012, 10.0246, 1),
                ('2015-01-01T00:00:00.000000000', 3.9725, 0.3973, 7.9450, 1),
                ('2015-02-01T00:00:00.000000000', 3.8694, 0.3869, 7.7387, 1),
                ('2015-03-01T00:00:00.000000000', 4.4411, 0.4441, 8.8822, 1),
                ('2015-04-01T00:00:00.000000000', 4.3822, 0.4382, 8.7643, 1),
                ('2015-05-01T00:00:00.000000000', 4.6262, 0.4626, 9.2523, 1),
                ('2015-06-01T00:00:00.000000000', 4.4823, 0.4482, 8.9646, 1),
                ('2015-07-01T00:00:00.000000000', 4.5771, 0.4577, 9.1542, 1),
                ('2015-08-01T00:00:00.000000000', 4.5634, 0.4563, 9.1268, 1),
                ('2015-09-01T00:00:00.000000000', 4.3092, 0.4309, 8.6183, 1),
                ('2015-10-01T00:00:00.000000000', 4.4496, 0.4450, 8.8992, 1),
                ('2015-11-01T00:00:00.000000000', 4.4451, 0.4445, 8.8901, 1),
                ('2015-12-01T00:00:00.000000000', 5.1825, 0.5183, 10.3651, 1),
                ('2016-01-01T00:00:00.000000000', 4.0093, 0.4009, 8.0186, 1),
                ('2016-02-01T00:00:00.000000000', 4.1355, 0.4136, 8.2711, 1),
                ('2016-03-01T00:00:00.000000000', 4.6009, 0.4601, 9.2019, 1),
                ('2016-04-01T00:00:00.000000000', 4.5094, 0.4509, 9.0187, 1),
                ('2016-05-01T00:00:00.000000000', 4.7142, 0.4714, 9.4284, 1),
            ],
            dtype=[
                ('ds', '<M8[ns]'),
                ('y', '<i8'),
                ('yhat_lower', '<i8'),
                ('yhat_upper', '<i8'),
                ('sign', '<i8'),
            ],
        ),
    )


def perturb_ts(df, col, scale=1):
    """Add noise to ts
    """
    mean = df[col].mean() * scale
    df[col] += np.random.default_rng(42).uniform(
        low=-mean / 2, high=mean / 2, size=len(df)
    )
    return df


@pytest.mark.mpl_image_compare
def test_create_forecast_figure(sample_data_df):
    time_series = sample_data_df.iloc[:30]
    predictions = sample_data_df.iloc[30:]
    predictions = predictions.rename(columns={Y_COL: YHAT_COL})
    full_series = pd.concat([predictions, time_series])
    full_series[DS_COL] = pd.to_datetime(full_series[DS_COL])
    end_date = pd.to_datetime(predictions[DS_COL]).min()
    forecast_window = (pd.to_datetime(predictions[DS_COL]).max() - end_date).days
    fig = plot.create_forecast_figure(
        full_series,
        'test',
        end_date,
        forecast_window,
        time_granularity=DAILY_TIME_GRANULARITY,
        plot_config=deepcopy(PLOT_CONFIG),
    )
    return fig


@pytest.mark.mpl_image_compare
def test_create_forecast_figure_overlapping(sample_data_yhat_df):
    time_series = sample_data_yhat_df
    predictions = sample_data_yhat_df.iloc[30:]
    predictions = predictions.rename(columns={Y_COL: YHAT_COL})
    predictions = perturb_ts(predictions, YHAT_COL, scale=0.1)
    full_series = pd.concat([predictions, time_series])
    full_series[DS_COL] = pd.to_datetime(full_series[DS_COL])
    end_date = pd.to_datetime(predictions[DS_COL]).min()
    forecast_window = (pd.to_datetime(predictions[DS_COL]).max() - end_date).days
    fig = plot.create_forecast_figure(
        full_series,
        'test',
        end_date,
        forecast_window,
        anomaly_window=0.5,
        time_granularity=HOURLY_TIME_GRANULARITY,
    )
    return fig
