"""Auxiliary functions for plotting"""
from typing import Dict, Optional, Tuple
from datetime import timedelta
from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from muttlib.plotting.constants import (
    ANOMALY_WIN,
    ANOMALY_WIN_FILL,
    COLORS,
    DAILY_TIME_GRANULARITY,
    DATE_FORMAT,
    DS_COL,
    FIG_SIZE,
    FORECAST,
    HISTORY,
    HISTORY_FILL,
    HOURLY_TIME_GRANULARITY,
    LABELS,
    OUTLIER_SIGN_COL,
    OUTLIERS_HISTORY,
    OUTLIERS_NEGATIVE,
    OUTLIERS_POSITIVE,
    PLOT_CONFIG,
    Y_COL,
    YHAT_COL,
    YHAT_LOWER_COL,
    YHAT_UPPER_COL,
)


def _create_common_series(
    df: pd.DataFrame, ds_col: str, start_date=None, end_date=None
) -> Tuple[pd.DataFrame, np.ndarray]:  # pylint:disable=unused-argument
    """Get series data for start/end date range.

    Parameters
    ----------
    df : pd.DataFrame
        Series Dataframe
    ds_col : str
        Series column
    start_date : [type], optional
        Series starts date, by default None
    end_date : [type], optional
        Series ends date, by default None

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray]
        Filtered values and dates.
    """
    window = df
    if start_date:
        window = window.query(f"@start_date <= {ds_col}")
    if end_date:
        window = window.query(f"{ds_col} <= @end_date")
    window_dates = window[ds_col].dt.to_pydatetime()
    return window, window_dates


def create_forecast_figure(
    df: pd.DataFrame,
    metric_name: str,
    end_date,
    forecast_window,
    anomaly_window: int = 0,
    time_granularity: str = DAILY_TIME_GRANULARITY,
    plot_config: Optional[Dict] = None,
) -> Figure:
    """Plot trend, forecast and anomalies with history, anomaly and forecast phases.

    Parameters
    ----------
    df : pd.DataFrame
        Series Dataframe
    metric_name : str
        Metric Name
    end_date :
        End_Date
    forecast_window :
        Forecast Window
    anomaly_window : int, optional
        Anomaly Window, by default 0
    time_granularity : str, optional
        Time Granularity, by default DAILY_TIME_GRANULARITY
    plot_config : Optional[Dict], optional
        Plot Config, by default None

    Returns
    -------
    Figure
        Forecast Figure
    """
    if plot_config is None:
        plot_config = PLOT_CONFIG
    plot_config_: dict = deepcopy(plot_config)["anomaly_plot"]

    plot_time_conf: dict = plot_config_[time_granularity]
    color_conf: dict = plot_config_[COLORS]
    label_conf: dict = plot_config_[LABELS]

    anomaly_start_date = end_date - timedelta(days=(anomaly_window))
    future_date = end_date + timedelta(days=forecast_window)

    # Note: we need to convert datetime values in series to pydatetime explicitly
    # See: https://stackoverflow.com/q/29329725/2149400
    history, history_dates = _create_common_series(df, DS_COL, end_date=future_date)
    anomaly_win, anomaly_win_dates = _create_common_series(
        df, DS_COL, start_date=anomaly_start_date, end_date=end_date
    )
    forecast, forecast_dates = _create_common_series(
        df, DS_COL, start_date=end_date, end_date=future_date
    )

    date_format = plot_time_conf[DATE_FORMAT]
    fig_size = plot_time_conf[FIG_SIZE]

    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(
        history_dates,
        history[Y_COL],
        color=color_conf[HISTORY],
        label=label_conf[HISTORY],
    )
    ax.plot(
        forecast_dates,
        forecast[YHAT_COL],
        ls="--",
        lw=2,
        color=color_conf[FORECAST],
        label=label_conf[FORECAST],
    )

    if YHAT_LOWER_COL in df and YHAT_UPPER_COL in df:
        ax.plot(
            anomaly_win_dates,
            anomaly_win[Y_COL],
            color=color_conf[ANOMALY_WIN],
            label=label_conf[ANOMALY_WIN].format(anomaly_window=anomaly_window),
        )
        ax.fill_between(
            history_dates,
            history[YHAT_LOWER_COL],
            history[YHAT_UPPER_COL],
            color=color_conf[HISTORY_FILL],
            alpha=0.2,
        )

        ax.fill_between(
            anomaly_win_dates,
            anomaly_win[YHAT_LOWER_COL],
            anomaly_win[YHAT_UPPER_COL],
            color=color_conf[ANOMALY_WIN_FILL],
            alpha=0.6,
        )

        ax.fill_between(
            forecast_dates,
            forecast[YHAT_LOWER_COL],
            forecast[YHAT_UPPER_COL],
            color=color_conf[FORECAST],
            alpha=0.2,
        )

        for _, o in df.query(
            f"{DS_COL} >= @history_dates.min() & {OUTLIER_SIGN_COL} != 0"
        ).iterrows():
            o_ds = o[f"{DS_COL}"]
            color = color_conf[OUTLIERS_HISTORY]
            o_label = ""
            if o_ds.to_pydatetime() in np.unique(anomaly_win_dates):
                color = (
                    color_conf[OUTLIERS_POSITIVE]
                    if o[OUTLIER_SIGN_COL] > 0
                    else color_conf[OUTLIERS_NEGATIVE]
                )
                o_label = label_conf["outlier"].format(date=f"{o_ds:{date_format}}")
            ax.plot_date(
                x=o_ds,
                y=o[Y_COL],
                marker="o",
                markersize=6,
                alpha=0.7,
                color=color,
                label=o_label,
            )

    ax.set_xlabel(label_conf["xlabel"])
    ax.set_xlim([history_dates.min(), forecast_dates.max()])

    base_10_scale = 0
    median_val = df[Y_COL].median()
    if median_val > 0:
        base_10_scale = int(np.log10(median_val))

    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, pos: f"{(y * (10 ** -base_10_scale)):g}")
    )
    base_10_scale_zeros = base_10_scale * "0"
    ax.set_ylabel(
        label_conf["ylabel"].format(
            metric_name=metric_name, base_10_scale_zeros=base_10_scale_zeros
        )
    )

    if anomaly_window > 0:
        plot_type = "Anomaly"
        start_date = anomaly_start_date
    else:
        plot_type = "Forecast"
        start_date = end_date + timedelta(days=1)
        end_date = future_date
    title = plot_config_["title"].format(
        plot_type=plot_type,
        metric_name=metric_name,
        start_date=start_date,
        end_date=end_date,
    )

    if time_granularity == HOURLY_TIME_GRANULARITY:
        title += f" {end_date:%H}hs"
    ax.set_title(title)
    ax.grid(True, which="major", c=color_conf["axis_grid"], ls="-", lw=1, alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    return fig
