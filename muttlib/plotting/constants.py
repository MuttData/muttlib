"""Plotting module constants"""

# Global
DS_COL = "ds"
Y_COL = "y"
YHAT_COL = "yhat"
OUTLIER_SIGN_COL = "sign"
YHAT_LOWER_COL = "yhat_lower"
YHAT_UPPER_COL = "yhat_upper"


# Time and Geo Granularity
HOURLY_TIME_GRANULARITY = "H"
DAILY_TIME_GRANULARITY = "D"
MONTHLY_TIME_GRANULARITY = "M"


# Plot common config keys
HISTORY = "history"
FORECAST = "forecast"
OUTLIER = "outlier"
COLORS = "colors"
ANOMALY_PLOT = "anomaly_plot"
LABELS = "labels"
ANOMALY_WIN = "anomaly_win"
ANOMALY_WIN_FILL = "anomaly_win_fill"
HISTORY_FILL = "history_fill"
OUTLIERS_HISTORY = "outliers_history"
OUTLIERS_POSITIVE = "outliers_positive"
OUTLIERS_NEGATIVE = "outliers_negative"
AXIS_GRID = "axis_grid"


# Daily and Hourly config
FIG_SIZE = "fig_size"
MAJOR_INTERVAL = "major_interval"
MINOR_INTERVAL = "minor_locator_interval"
FUTURE_WINDOW = "future_window"
HISTORY_WINDOW = "history_window"
FONT_SIZE = "font_size"
MINOR_LABEL_ROTATION = "minor_labelrotation"
MAJOR_PAD = "major_pad"
DATE_FORMAT = "date_format"


# Plots config
PLOT_CONFIG = {
    ANOMALY_PLOT: {
        HOURLY_TIME_GRANULARITY: {
            FIG_SIZE: (13, 9),
            MAJOR_INTERVAL: 2,
            FUTURE_WINDOW: 5,
            HISTORY_WINDOW: 14,
            FONT_SIZE: 7,
            MINOR_LABEL_ROTATION: 40,
            MAJOR_PAD: 25,
            MINOR_INTERVAL: 8,
            DATE_FORMAT: "%Y-%b-%d %Hhs",
        },
        DAILY_TIME_GRANULARITY: {
            FIG_SIZE: (10, 6),
            MAJOR_INTERVAL: 3,
            FUTURE_WINDOW: 15,
            HISTORY_WINDOW: 60,
            FONT_SIZE: 7,
            DATE_FORMAT: "%Y-%b-%d",
        },
        MONTHLY_TIME_GRANULARITY: {
            FIG_SIZE: (13, 9),
            MAJOR_INTERVAL: 2,
            FUTURE_WINDOW: 5,
            HISTORY_WINDOW: 14,
            FONT_SIZE: 7,
            MINOR_LABEL_ROTATION: 40,
            MAJOR_PAD: 25,
            MINOR_INTERVAL: 8,
            DATE_FORMAT: "%Y-%b",
        },
        COLORS: {
            HISTORY: "orange",
            HISTORY_FILL: "gray",
            ANOMALY_WIN: "dodgerblue",
            ANOMALY_WIN_FILL: "lightskyblue",
            FORECAST: "blue",
            OUTLIERS_HISTORY: "black",
            OUTLIERS_POSITIVE: "green",
            OUTLIERS_NEGATIVE: "red",
            AXIS_GRID: "gray",
        },
        LABELS: {
            "xlabel": "Date",
            "ylabel": "{metric_name} ({base_10_scale_zeros}s)",
            HISTORY: "History",
            ANOMALY_WIN: "Last {anomaly_window} days",
            FORECAST: "Forecast",
            OUTLIER: "Outlier: {date}",
        },
        "title": "{plot_type} for {metric_name} from {start_date:%d-%b} to {end_date:%d-%b}.",
    }
}
