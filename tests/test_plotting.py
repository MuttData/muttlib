from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from muttlib.plotting import plot
from muttlib.plotting.constants import (
    DAILY_TIME_GRANULARITY,
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


def perturb_ts(df, col, scale=1):
    """Add noise to ts
    """
    mean = df[col].mean() * scale
    df[col] += np.random.default_rng(42).uniform(
        low=-mean / 2, high=mean / 2, size=len(df)
    )
    return df


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


def test_create_forecast_figure_overlapping(sample_data_df):
    time_series = sample_data_df
    predictions = sample_data_df.iloc[30:]
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
        time_granularity=DAILY_TIME_GRANULARITY,
        plot_config=deepcopy(PLOT_CONFIG),
    )


def test_ForecastPlotterTask_overlapping(sample_data_df):
    time_series = sample_data_df
    predictions = sample_data_df.iloc[30:]
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
        time_granularity=DAILY_TIME_GRANULARITY,
        plot_config=deepcopy(PLOT_CONFIG),
    )
