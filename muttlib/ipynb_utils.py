"""Helper functions for ipyb visualizations and explorations."""
import re
import _string
import subprocess
from hashlib import md5
from functools import partial
from pathlib import Path
from string import Formatter

import pandas as pd
import numpy as np
from tabulate import tabulate
from typing import List
from jinja2 import Undefined
from jinjasql import JinjaSql
from IPython.display import display
import matplotlib
# Special back-end set to have the ipynb **not** use tkinter
matplotlib.use('Agg')
import matplotlib.pyplot as plt # NOQA
# For nice df prints that can be copy pasted to chat services
import seaborn as sns # NOQA

# Cleanear matplotlib dates as day in letters
import matplotlib.dates as mdates
# Cleanear matplotlib formatting
from matplotlib import ticker

NULL_COUNT_CLAUSE = """SUM( CASE WHEN {col} IS NULL
    THEN 1 ELSE 0 END ) AS {as_col}"""


def convert_to_snake_case(name: str):
    """Convert string to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def tabulated_df(df: pd.DataFrame):
    """Pretty print, ready-to-clipboard dataframes in ipynbs."""
    print(tabulate(df, headers='keys', tablefmt='psql'))


def list_to_sql_tuple(l: List)->str:
    """Create an sql-string synthax-valid tuple from python list."""
    assert len(l) > 0
    placeholders = ', '.join(str(element) for element in l)
    return f'({placeholders:s})'


def describe_table(table_name: str, db_connector)->pd.DataFrame:
    """Describe table sql template."""
    desc = db_connector.execute(f'describe {table_name}')
    return desc


def write_to_clipboard(output)->None:
    """Write str to clipboard using UTF-8 encoding."""
    process = subprocess.Popen(
        'pbcopy', env={'LANG': 'en_US.UTF-8'}, stdin=subprocess.PIPE)
    process.communicate(output.encode('utf-8'))


def list_vals_contains_str(in_list, val: str):
    """Filter string list containing certain string match in lowercase."""
    return [s for s in in_list if val.lower() in s.lower()]


def ab_split(id, salt, control_group_size: float):
    """Return True (for test) or False (for control).

    Logic is based on the ID string and salt.. The control_group_size number
    is between 0 and 1. This sets how big the control group is in perc.
    """
    test_id = str(id) + '-' + str(salt)
    test_id_digest = md5(test_id.encode('ascii')).hexdigest()
    test_id_first_digits = test_id_digest[:6]
    test_id_last_int = int(test_id_first_digits, 16)
    ab_split = (test_id_last_int/0xFFFFFF)
    return ab_split > control_group_size


def get_ordered_category_levels(df, cat_col, top_n=None):
    """
    Return a list of a categorical column's levels and num of levels.

    Levels list are ordered by descending popularity.
    """
    rv = df[cat_col].value_counts().index[:top_n]
    return rv, len(rv)


def col_sample_display(df: pd.DataFrame, col: str,
                       quantile: float=None, top_val: float=None):
    """Fast printing/visualization of sample data for given column.

    Also shows 10 unique specific values from the column and has
    modifiers for either showing a histogram for numeric data, or
    showing top_value counts for non-numeric columns.
    """
    unique_vals = df[col].unique()
    null_count = df[col].isnull().sum()
    null_pct = null_count/df.shape[0]
    print(f'\nCol is {col}\n')
    print(f'Null count is {null_count}, Null percentage is: {null_pct:.2%}')
    print(len(unique_vals), unique_vals[0:10],)
    display(df[col].describe())
    display(df[col].sample(10))

    # check either numerical or not
    if not np.issubdtype(df[col].dtype, np.number):

        val_counts = df[col].value_counts().to_frame()
        val_counts['percentage'] = 100*val_counts[col]/val_counts[col].sum()
        display(val_counts.head(10))
    else:

        query_str = f'{col}== {col}'
        if quantile is not None:
            top_perc = df[col].quantile(q=quantile)
            # this +100 is a safety net for when top_perc results
            # are equal to the lower limit of the filter.
            query_str = f'{col}>=0 and {col}<= {top_perc+100}'

        elif top_val is not None:
            query_str = f'{col}<= {top_val}'

        df.query(query_str)[col].hist(bins=60)
        plt.title(query_str)


def top_categorical_vs_kdeplot(
        df: pd.DataFrame,
        categorical_col: str,
        numerical_col: str,
        quantile: float=None,
        upper_bound_val: float = None,
        num_category_levels: int=2):
    """
    Plot multiple kdeplots for each category level.

    We would like to plot different kdeplots for a numerical column,
    yet generating a different plot for all rows corresponding to
    a specific category level (the possible values of the category set).
    """
    # Get enough colors for our test
    palette = sns.color_palette("husl", num_category_levels)
    top_values, _ = get_ordered_category_levels(df,
                                                categorical_col,
                                                num_category_levels)

    # Default query to filter data
    query_str = f'{numerical_col}=={numerical_col}'
    if quantile is not None:
        top_perc = df[numerical_col].quantile(q=quantile)
        # This +10 is a safety net for when top_perc results
        # Are equal to the lower limit of the filter.
        query_str = f'{numerical_col}>=0 and {numerical_col}<= {top_perc}'

    elif upper_bound_val is not None:
        query_str = f'{numerical_col}<= {upper_bound_val}'

    # Filter data
    view = df.query(query_str)
    i = 0
    # Create grouping condition on top category values
    gr_condition = (view[categorical_col]
                    .where(view[categorical_col].isin(top_values))
                    )

    # Group and plot for each category level
    iterator = view.groupby(gr_condition)[numerical_col]
    for name, grp in iterator:
        sns.kdeplot(grp, shade=True, alpha=.4,
                    label=f'{name}', color=palette[i])
        i = i+1
    title_str = f"Feature {numerical_col} distributions across "
    title_str += f"different {categorical_col}"
    plt.title(title_str, fontsize=15)
    plt.xlabel(f'{numerical_col} value')
    plt.ylabel('Probability')
    plt.show()


def top_categorical_vs_heatmap(
    df: pd.DataFrame,
    dependent_col: str,
    ind_col: str, quantile: float=None,
    top_val: float = None,
    with_log: bool=False
):
    """
    Plot heatmap from two variables which are categorical.

    Where one variable is categorical and in the index, and the other
    is categorical but is the dependent variable, as columns.
    We generate a plot for all values/levels of the index.
    """
    # Default query to filter data
    query_str = f'{dependent_col}== {dependent_col}'
    if quantile is not None:
        top_perc = df[ind_col].quantile(q=quantile)
        query_str = f'{ind_col}<= {top_perc}'

    elif top_val is not None:
        query_str = f'{ind_col}<= {top_val}'

    # Filter data
    view = df.query(query_str)

    # Create pivot with top category values
    agg_fun = (lambda x: np.log(x+1)) if with_log else (lambda x: x)
    gr_cols = [dependent_col, ind_col]
    pivot_grid = (view[gr_cols].groupby(gr_cols).size().apply(agg_fun)
                  .reset_index()
                  .pivot(index=ind_col,
                         columns=dependent_col,
                         values=0))
    # Normalize vals per column
    max_min_range = pivot_grid.max()-pivot_grid.min()
    pivot_grid = (pivot_grid-pivot_grid.min()) / max_min_range
    sns.heatmap(pivot_grid)

    title_str = f'Heatmap of  {ind_col} level percentages across '
    title_str += f'different {dependent_col}'
    plt.title(title_str, fontsize=15)
    plt.xlabel(f'{dependent_col} value')
    plt.ylabel(f'{ind_col} level')
    plt.show()


def get_one_to_one_relationship(
    df: pd.DataFrame,
    factor_id: str,
    factor_name: str
):
    """Do for a given factor, which we understand as a categorical column.

    Of different category levels. We would like to know if there is a
    1-1 relation between the ids of the values of that factor with the
    column corresponding to the names of those ids.
    """
    rv = None
    g = df[[factor_id, factor_name]].groupby(factor_id)
    id_col_name_counts = g.transform(lambda x: len(x.unique()))
    rv = id_col_name_counts
    return rv


def sum_count_aggregation(
    df: pd.DataFrame,
    group_cols: List,
    aggregation_operations: List=['sum', 'count'],
    numerical_cols: List=['PONDERACION_RIESGO_PAYMENT', 'Q_RECARGA']
):
    """Aggregate data by a gruop of columns into sum and count."""
    # Create aggregating dictionary
    agg_dict = {col: aggregation_operations for col in numerical_cols}

    # Group and aggregate
    counts = df.groupby(group_cols).agg(agg_dict)

    # Flatten multi-hierarchy index
    counts.columns = ['_'.join(col).strip() for col in counts.columns.values]

    for col in numerical_cols:
        for aggr in aggregation_operations:
            perc_col = '_'.join([col, aggr, 'perc'])
            aggr_col = '_'.join([col, aggr])
            counts[perc_col] = counts[aggr_col]/counts[aggr_col].sum()

    # Chose one column to sort for [first column]
    sort_col = [col for col in counts.columns if 'count' in col][0]
    counts.sort_values(by=sort_col, ascending=False)
    return counts


def sum_count_time_series(
    df: pd.DataFrame,
    date_col: str,
    resample_frequency: str='D',
    aggregation_operations: List=['sum', 'count'],
    numerical_series: List=['PONDERACION_RIESGO_PAYMENT', 'Q_RECARGA'],
    filter_query: str=None  # to select a subset of the whole database only
):
    """Get time series grouping by a certain time-window.

    Only for a view of the original df.
    """
    if not filter_query:
        filter_query = 'advertiser_name == advertiser_name'
    # generate aggregating dictionary
    agg_dict = {col: aggregation_operations for col in numerical_series}

    # Count the amount of events in this time frequency
    time_series = (df.query(filter_query).resample(
        resample_frequency,
        on=date_col)[numerical_series].agg(agg_dict)
        )

    # Flatten multi-hierarchy index
    time_series.columns = [
        '_'.join(col).strip()
        for col in time_series.columns.values
    ]
    # Reset index and sort by oldest event date first
    time_series = time_series.reset_index().sort_values(date_col)

    return time_series


def plot_agg_bar_charts(
    agg_df,
    agg_ops,
    group_cols,
    series_col: str='impressions',
    perc_filter: float=0.03
):
    """Plot bar charts on an aggregated dataframe."""
    perc_cols = ['_'.join([series_col, aggr, 'perc']) for aggr in agg_ops]

    fig = plt.figure()
    # Remove small levels in perc.
    (agg_df.query(f'{perc_cols[0]}>@perc_filter')[perc_cols].T.plot
        .bar(
            stacked=True, figsize=(8, 6),
            colormap="GnBu", label='right',
            ax=fig.gca(), rot=0))

    title_str = f'Grouped by:{group_cols!s}, aggregated by: {agg_ops!s}, '
    title_str += f'filter at least {perc_filter} perc'
    plt.title(title_str, color='black')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.show()


def plot_category2category_pie_charts(
        df,
        cat_col,
        cat2_col,
        max_category_levels=4,
        n_rows=1,
        figsize=(16, 6),
        autopct='%.1f',
        sample_frac=0.1,
):
    """
    Plot a single pie-chart per 1st catcol. Display % sizes of 2nd catcol.

    The maximum category levels will filter out both categorical's col "tail"
    levels.
    """
    unique_levels, n_cols = get_ordered_category_levels(df, cat_col)
    unique2_levels, _ = get_ordered_category_levels(df, cat2_col)

    if max_category_levels < len(unique_levels):
        col_name = f'reduced_{cat_col}'
        df[col_name] = category_reductor(
            df, cat_col, n_levels=max_category_levels)
        cat_col = col_name  # update new category col
        unique_levels, n_cols = get_ordered_category_levels(df, cat_col)

    if max_category_levels < len(unique2_levels):
        col_name = f'reduced_{cat2_col}'
        df[col_name] = category_reductor(
            df, cat2_col, n_levels=max_category_levels)
        cat2_col = col_name  # update new category col
        unique2_levels, _ = get_ordered_category_levels(df, cat2_col)

    # Plot pie charts
    fig, axes = plt.subplots(figsize=figsize)

    # Remove unnecessary bounding box lines, xticks, yticks etc.
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.tick_params(
        top=False,
        bottom=False,
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False)

    for i, level in enumerate(unique_levels):
        subplot_num = n_rows * 100 + n_cols * 10 + i + 1
        ax = fig.add_subplot(subplot_num)
        ax.set_title(level)
        ax.axis('off')  # remove unnecessary figure axis
        v = df.query(f'{cat_col} == @level')
        ratio_percentage = v[cat2_col].dropna().sample(
            frac=sample_frac).value_counts(
            normalize=True) * 100

        # Save first df to mantain same color order for all subplots
        if i == 0:
            first_ix = ratio_percentage.index
        elif i > 0:
            ratio_percentage = ratio_percentage.reindex(first_ix).dropna()

        # Subplot pie
        ratio_percentage.plot.pie(
            autopct=autopct,
            legend=True,
            ax=ax,
            title=level,
        )

def plot_timeseries(
    df,
    y_col,
    fig_size=(10, 8),
    non_index_col=None,
    title_str="",
    hourly_formatted=False,
    fig_ax=None,
    fmt='-o',
    label=None,
    color=None,
    **kwargs,
):
    """
    Create a special tseries plot from a df's index and y-col.

    Various daily and hourly formats for xticks can be used.
    Optionally one can pass a specific column to act as index.

    Parameters
    ----------
    df : pd.DataFrame
        At least one values column (y) and one date-kind index
        or column with ordinality.
    y_col : str
        The name of the colum with the values.
    fig_size : int tuple, optional
        The x,y size cm size for the figure.
    non_index_col : str, optional
        By default the plot will use the df index as the ordinal column,
        yet if passed, this arg will be the key of the col to be used as
        index (x-axis).
    title_str : str, optional
        Used as title for the figure.
    hourly_formatted : bool, optional
        If major/minor xticks should be formatted for hourly-kind of data.
        It may clutter the axis when using data with long time-ranges.
    fig_ax : tuple, optional
        A matplotlib (fig, ax) tuple that can be used as base for this
        plot.
    fmt : [matplotlib] str, optional
        The series's format string.
    label : [matplotlib] str, optional
        The series's legend name.
    color : [matplotlib] str, optional
        The series color.
    kwargs: dict, optional
        Additional matplotlib-kind of arguments passed to the matplotlib
        plotting functions.

    Returns
    -------
    A matplotlib (figure, axis) tuple.
    """
    if not fig_ax:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax
    indext = df.index if not non_index_col else df[non_index_col]
    ix_date_type = np.issubdtype(indext.dtype, np.datetime64)

    median_y_val = df[y_col].quantile(0.5)
    # Plot values on index
    plot_method = ax.plot_date if ix_date_type else ax.plot
    plot_method(indext, df[y_col], fmt=fmt, label=label, color=color, **kwargs)

    if ix_date_type:

        # Ticks formatting
        monthly_format = mdates.DateFormatter('\n\n\n\n\n%b\n%Y')
        daily_format = mdates.DateFormatter('\n\n%d\n%a')
        hourly_format = mdates.DateFormatter('%Hhs\n%a')
        # Ticks locations
        hourly_locator = mdates.HourLocator(byhour=range(0, 24, 3), interval=5)
        dow_locator = mdates.WeekdayLocator(byweekday=(0), interval=1)
        month_locator = mdates.MonthLocator()

        # set minor xaxis formatting
        min_locator_obj = hourly_locator if hourly_formatted else dow_locator
        min_format_obj = hourly_format if hourly_formatted else daily_format
        ax.xaxis.set_minor_locator(min_locator_obj)
        ax.xaxis.set_minor_formatter(min_format_obj)

        # set major xaxis formatting
        maj_locator_obj = dow_locator if hourly_formatted else month_locator
        maj_format_obj = daily_format if hourly_formatted else monthly_format
        ax.xaxis.set_major_locator(maj_locator_obj)
        ax.xaxis.set_major_formatter(maj_format_obj)

        # Add title and subtitle
        min_date = indext.min().date()
        max_date = indext.max().date()
        metadata_str = f"Data from {min_date} thru {max_date}"
        plt.title(metadata_str)

    ax.xaxis.grid(True, which="minor")
    ax.yaxis.grid()

    if median_y_val > 1000:
        # Add major y axis formatting for thousands
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, p: format(int(x), ','))
        )
    fig.set_size_inches(fig_size)
    plt.suptitle(wrap(title_str))
    ax.set_ylabel(y_col)
    return fig, ax


def category_reductor(df, categorical_col, n_levels=8,
                      default_level='Other'):
    """Reduce a categorical col's levels.

    This outputs a new cat col with reduced levels.
    It will not modify any null values in original category.
    """
    top_levels, _ = get_ordered_category_levels(df, categorical_col,
                                                n_levels - 1)

    def sub_categorize(x):
        """Reduce category series levels."""
        if x in top_levels:
            return x
        else:
            return default_level

    # Modify only non-null values
    rv = df[categorical_col].apply(
        lambda x: sub_categorize(x) if (pd.notnull(x)) else x)

    return rv


def clean_numeric_col(df, numeric_col, **kwds):
    """Clean/cast a numeric.column from object/string type.

    This outputs a new series with numeric or null values.
    """
    # Remove emtpy whitespace
    df[numeric_col] = df[numeric_col].str.replace(' ', '').replace('', np.nan)
    # Convert to float. Defauls pushing errors as nulls
    return partial(pd.to_numeric, errors='raise')(df[numeric_col], **kwds)


def optimize_numeric_types(df):
    """Cast numeric columns to more memory friendly types."""
    # Check which types best.
    new_ftypes = {
        c: 'float32'
        for c in df.dtypes[df.dtypes == 'float64'].index
    }
    df = df.astype(new_ftypes, copy=False)
    return df


def get_string_named_placeholders(s):
    """Output list of placeholders in a formatted string."""
    rv = [_string.formatter_field_name_split(fn)[0] for _, fn, _, _
          in Formatter().parse(s) if fn is not None]
    return rv


def load_sql_query(sql, query_context_params=None):
    """Read sql file or string and format with a dictionary of params."""
    pat = Path(sql).expanduser()
    # This `pat.is_file()` breaks for long enough
    if np.DataSource().exists(pat.as_posix()):
        with open(pat, 'r') as f:
            sql = f.read()

    if query_context_params:
        j = JinjaSql(param_style='pyformat')
        binded_sql, bind_params = j.prepare_query(sql,
                                                  query_context_params)
        missing_placeholders = [
            k for k, v in bind_params.items() if Undefined() == v
        ]

        assert len(missing_placeholders) == 0, (
            f"Missing placeholders are: {missing_placeholders}")

        try:
            sql = binded_sql % bind_params
        except KeyError as e:
            print(e)
            return

    return sql


def get_sql_stats_aggr(input_expression, as_name=None, with_std=False,
                       with_ndv=False, with_count=False):
    """Get Cloudera-valid battery of statistical aggregations clause."""
    rv = f"""
    SUM({input_expression}) as sum_{as_name},
    AVG({input_expression}) as mean_{as_name},
    APPX_MEDIAN({input_expression}) as median_{as_name},
    MIN({input_expression}) as min_{as_name},
    MAX({input_expression}) as max_{as_name},"""

    if with_std:
        rv += f"\n STDDEV({input_expression}) as std_{as_name},"
    if with_ndv:
        rv += f"\n NDV({input_expression}) as unique_{as_name},"
    if with_count:
        rv += f'\n COUNT(1) as count_{as_name},'

    return rv


def get_null_count_aggr(columns_list,
                        as_name="null_count_",
                        no_ending_comma=False,
                        empty_string_null=False):
    """Get Cloudera-valid expression counting nulls for columns."""
    rv = ""
    pre_clause = NULL_COUNT_CLAUSE
    if empty_string_null:

        pre_clause = pre_clause.replace('IS NULL', "= ''")
    for col in columns_list:

        rv += pre_clause.format(col=col, as_col=as_name + col) + ",\n"
    if no_ending_comma:

        rv = rv.rsplit(',', 1)[0]

    return rv


def get_include_exclude_columns(
    cols, include_regexes=None, exclude_regexes=None
):
    """Filter list by inclusion and exclusion regexes."""
    if include_regexes is None:
        ret = cols
    else:
        ret = get_matching_columns(cols, include_regexes)
    ret = set(ret)
    if exclude_regexes:
        ret.difference_update(get_matching_columns(cols, exclude_regexes))
    return sorted(list(ret))


def get_matching_columns(cols, regex_list):
    """Match a list of columns with a nuber of regexes."""
    ret = []
    for regex in regex_list:
        regex = re.compile(regex)
        ret += filter(regex.search, cols)
    return ret
    

def get_sqlserver_hashed_sample_clause(id_clause, sample_pct):
    """Get SQL Server-valid synthax for hashed-sampling an id clause.on

    Takes as imput a given sample_pct in (0, 1)."""
    assert 0 < sample_pct < 1, f"{sample_pct} should be a float  in (0,1)"
    int_pct = int(sample_pct * 100)
    rv = f"""
    AND ABS(CAST(HASHBYTES('SHA1', 
        {id_clause}) AS BIGINT)) % 100 <= {int_pct}"""
    return rv

def wrap_list_values_quotes(lis):
    """Wraps all values in a list with single quotes."""
    return [f"'{val}'" for val in lis]
