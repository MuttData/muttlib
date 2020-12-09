"""Helper functions for ipynb visualizations and explorations."""
import re
import _string
import subprocess
from hashlib import md5
import logging
from functools import partial
from pathlib import Path
from string import Formatter

import pandas
import numpy as np
from typing import List
from jinja2 import Undefined
from jinjasql import JinjaSql
from IPython.display import display

import matplotlib

import muttlib.utils as utils

logger = logging.getLogger(f'ipynb_utils.{__name__}')

# Special back-end set to have the ipynb **not** use tkinter
current_backend = matplotlib.get_backend()
if current_backend != 'TkAgg':
    logger.warning(
        f"You are currently using {current_backend} as your Matplotlib backend. "
        "This lib currently recommends using TkAgg for correctly visualizing its plots."
    )
import matplotlib.pyplot as plt  # NOQA


NULL_COUNT_CLAUSE = """SUM( CASE WHEN {col} IS NULL
    THEN 1 ELSE 0 END ) AS {as_col}"""


def convert_to_snake_case(raw_string: str):
    """Convert string to snake_case.

    Parameters
    ----------
    raw_string: str
        Raw string to convert.

    Returns
    -------
        String converted to snake_case: str

    Examples
    --------
        >>> convert_to_snake_case('Batman-and-Robin')
        ... 'batman_and_robin'

        >>> convert_to_snake_case('RobinAndBatman')
        ... 'robin_and_batman'
    """
    raw_string = re.sub(r' +|-|_', r' ', raw_string)
    raw_string = raw_string.strip()
    if len(raw_string.split(' ')) > 1:
        raw_string = ''.join([w.capitalize() for w in raw_string.split(' ')])
    raw_string = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', raw_string)
    raw_string = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', raw_string).lower()
    return raw_string


def list_to_sql_tuple(l: List) -> str:
    """Create an sql-string synthax-valid tuple from python list."""
    assert len(l) > 0
    placeholders = ', '.join(str(element) for element in l)
    return f'({placeholders:s})'


def describe_table(table_name: str, db_connector) -> pandas.DataFrame:
    """Describe table sql template.

    Parameters
    ----------
    table_name: str

    db_connector:


    Returns
    -------

    """
    desc = db_connector.execute(f'describe {table_name}')
    return desc


def write_to_clipboard(output) -> None:
    """Write str to clipboard using UTF-8 encoding.

    Parameters
    ----------
    output :


    Returns
    -------

    """
    process = subprocess.Popen(
        'pbcopy', env={'LANG': 'en_US.UTF-8'}, stdin=subprocess.PIPE
    )
    process.communicate(output.encode('utf-8'))


def list_vals_contains_str(value_list: list, filter_value: str) -> list:
    """Filter string list containing certain string match in lowercase.

    Parameters
    ----------
    value_list: list
        List with string values

    filter_value: str
        String filter_value to filter

    Returns
    -------
        List filtered by _filter_value_ : list

    Examples
    --------
        >>> filter_value = 'batman'
        >>> value_list = ['robin', 'batman', 'riddler', 'two_face', 'Batman']
        >>> assert list_vals_contains_str(value_list, filter_value)
        ... ['batman', 'Batman']

    """
    return [s for s in value_list if filter_value.lower() in s.lower()]


def ab_split(id_obj: str, salt: str, control_group_size: float):
    """Split object into test or control group based on the ID and salt.

    Parameters
    ----------
    id_obj : str
        Object id.
    salt : str
        Salt value.
    control_group_size : float
        Sets how big the control group is desired in percentage.
        Must be between 0 and 1.

    Returns
    -------
        True (for test) or False (for control) : bool

    Examples
    --------
        >>> users = pandas.DataFrame({'id': numpy.arange(100**2)})
        >>> users['test_group'] = users.id.apply(
        ...     lambda id: ab_split(id, 'E1F53135E559C253', 0.25))
        >>> users.test_group.value_counts(normalize=True)
        True     0.7465
        False    0.2535
        Name: test_group, dtype: float64
    """
    test_id = str(id_obj) + '-' + str(salt)
    test_id_digest = md5(test_id.encode('ascii')).hexdigest()  # nosec
    test_id_first_digits = test_id_digest[:6]
    test_id_last_int = int(test_id_first_digits, 16)
    split = test_id_last_int / 0xFFFFFF
    return split > control_group_size


def col_sample_display(
    df: pandas.DataFrame,
    col: str,
    quantile: float = None,
    top_val: float = None,
    num_sample: float = 300,
):
    """Fast printing/visualization of sample data for given column.

    Also shows 10 unique specific values from the column and has
    modifiers for either showing a histogram for numeric data, or
    showing top_value counts for non-numeric columns.

    Parameters
    ----------
    df: pandas.DataFrame :

    col: str :

    quantile: float :
         (Default value = None)
    top_val: float :
         (Default value = None)
    num_sample: float :
         (Default value = 300)

    Returns
    -------

    """
    len_col = df[col].shape[0]
    unique_vals = df[col].unique()
    num_unique_vals = len(unique_vals)
    null_count = df[col].isnull().sum()
    null_pct = null_count / df.shape[0]
    print(f'\nCol is {col}\n')
    print(f'Null count is {null_count}, Null percentage is: {null_pct:.2%}')
    print(num_unique_vals, unique_vals[0:10])
    display(df[col].describe())
    display(df[col].sample(10))

    # Various checks either numerical or not
    if len_col < num_sample:
        num_sample = len_col
    if col in utils.df_get_typed_cols(df, 'date'):
        # Direct exit with `date`, as they can be miscasted to numeric
        is_numeric_type = False
    else:
        if col in utils.df_get_typed_cols(df, 'num'):
            is_numeric_type = True
        else:  # cols that are string which might get converted
            try:
                pandas.to_numeric(df[col].sample(num_sample))
                is_numeric_type = True
            except ValueError:
                is_numeric_type = False

    if is_numeric_type or num_unique_vals < 15:
        val_counts = df[col].value_counts().to_frame()
        val_counts.index.name = col
        val_counts.rename(columns={col: 'count'}, inplace=True)
        val_counts['percentage'] = 100 * val_counts['count'] / val_counts['count'].sum()
        display(val_counts.head(10))

    if is_numeric_type:
        num_col = df[[col]].copy()  # keep in dataframe form to use the .query method
        num_col[col] = pandas.to_numeric(num_col[col].values, errors='coerce')
        query_str = f'{col} == {col}'
        if quantile is not None:
            top_perc = num_col[col].quantile(q=quantile)
            # this +100 is a safety net for when top_perc results
            # are equal to the lower limit of the filter.
            query_str = f'{col}>=0 and {col}<= {top_perc+100}'

        elif top_val is not None:
            query_str = f'{col}<= {top_val}'

        num_col.query(query_str)[col].hist(bins=60)
        plt.title(query_str)


def get_one_to_one_relationship(df: pandas.DataFrame, factor_id: str, factor_name: str):
    """Do for a given factor, which we understand as a categorical column.

    Of different category levels. We would like to know if there is a
    1-1 relation between the ids of the values of that factor with the
    column corresponding to the names of those ids.

    Parameters
    ----------
    df: pandas.DataFrame :

    factor_id: str :

    factor_name: str :


    Returns
    -------

    """
    rv = None
    g = df[[factor_id, factor_name]].groupby(factor_id)
    id_col_name_counts = g.transform(lambda x: len(x.unique()))
    rv = id_col_name_counts
    return rv


def sum_count_aggregation(
    df: pandas.DataFrame,
    group_cols: List,
    numerical_cols: List,
    aggregation_operations=('sum', 'count'),
):
    """Aggregate data by a gruop of columns into sum and count.

    Parameters
    ----------
    df: pandas.DataFrame :

    group_cols: list :

    numerical_cols: list :

    aggregation_operations: tuple or list :
        (Default value = ('sum', 'count')) :

    Returns
    -------

    """
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
            counts[perc_col] = counts[aggr_col] / counts[aggr_col].sum()

    # Chose one column to sort for [first column]
    sort_col = [col for col in counts.columns if 'count' in col][0]
    counts.sort_values(by=sort_col, ascending=False)
    return counts


def sum_count_time_series(
    df: pandas.DataFrame,
    date_col: str,
    numerical_series: List,
    resample_frequency: str = 'D',
    aggregation_operations=('sum', 'count'),
    filter_query: str = None,  # to select a subset of the whole database only
):
    """Get a time series grouping in a a certain time-window.

    Only for a view of the original df.

    Parameters
    ----------
    df: pandas.DataFrame :

    date_col: str :

    numerical_series: list :

    resample_frequency: str :
         (Default value = 'D')
    aggregation_operations: tuple or list :
         (Default value = ('sum', 'count') :

    filter_query: str :
         (Default value = None)
    # to select a subset of the whole database only :


    Returns
    -------

    """
    if not filter_query:
        filter_query = f'{date_col} == {date_col}'
    # generate aggregating dictionary
    agg_dict = {col: aggregation_operations for col in numerical_series}

    # Count the amount of events in this time frequency
    time_series = (
        df.query(filter_query)
        .resample(resample_frequency, on=date_col)[numerical_series]
        .agg(agg_dict)
    )

    # Flatten multi-hierarchy index
    time_series.columns = ['_'.join(col).strip() for col in time_series.columns.values]
    # Reset index and sort by oldest event date first
    time_series = time_series.reset_index().sort_values(date_col)

    return time_series


def category_reductor(df, categorical_col, n_levels=8, default_level='Other'):
    """Reduce a categorical col's levels.

    This outputs a new cat col with reduced levels.
    It will not modify any null values in original category.

    Parameters
    ----------
    df :

    categorical_col :

    n_levels :
         (Default value = 8)
    default_level :
         (Default value = 'Other')

    Returns
    -------

    """
    top_levels, _ = utils.get_ordered_factor_levels(df, categorical_col, n_levels - 1)

    def sub_categorize(x, top_levels):
        """Reduce category series levels.

        Parameters
        ----------
        x :

        top_levels :


        Returns
        -------

        """
        if x in top_levels:
            return x
        else:
            return default_level

    # Modify only non-null values
    rv = df[categorical_col].apply(
        lambda x: sub_categorize(x, top_levels) if (pandas.notnull(x)) else x
    )

    return rv


def clean_numeric_col(df, numeric_col, **kwds):
    """Clean/cast a numeric.column from object/string type.

    This outputs a new series with numeric or null values.

    Parameters
    ----------
    df :

    numeric_col :

    **kwds :


    Returns
    -------

    """
    # Remove emtpy whitespace
    df[numeric_col] = df[numeric_col].str.replace(' ', '').replace('', np.nan)
    # Convert to float. Defauls pushing errors as nulls
    return partial(pandas.to_numeric, errors='raise')(df[numeric_col], **kwds)


def optimize_numeric_types(df: pandas.DataFrame):
    """Cast numeric columns to more memory friendly types.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------

        df : pandas.DataFrame
    """
    # Check which types best.
    new_ftypes = {c: 'float32' for c in df.dtypes[df.dtypes == 'float64'].index}
    df = df.astype(new_ftypes, copy=False)
    return df


def get_string_named_placeholders(s):
    """Output list of placeholders in a formatted string.

    Parameters
    ----------
    s : str

    Returns
    -------
        list : list
    """
    rv = [
        _string.formatter_field_name_split(fn)[0]
        for _, fn, _, _ in Formatter().parse(s)
        if fn is not None
    ]
    return rv


def load_sql_query(sql, query_context_params=None):
    """Read sql file or string and format with a dictionary of params.

    Parameters
    ----------
    sql :

    query_context_params :
         (Default value = None)

    Returns
    -------

    """
    pat = Path(sql).expanduser()
    if pat.exists():
        with open(pat, 'r') as f:
            sql = f.read()

    if query_context_params:
        j = JinjaSql(param_style='pyformat')
        binded_sql, bind_params = j.prepare_query(sql, query_context_params)
        missing_placeholders = [k for k, v in bind_params.items() if Undefined() == v]

        assert (
            len(missing_placeholders) == 0
        ), f'Missing placeholders are: {missing_placeholders}'

        try:
            sql = binded_sql % bind_params
        except KeyError as e:
            print(e)
            return

    return sql


def get_sql_stats_aggr(
    input_expression, as_name=None, with_std=False, with_ndv=False, with_count=False
):
    """Get Cloudera-valid battery of statistical aggregations clause.

    Parameters
    ----------
    input_expression :

    as_name :
         (Default value = None)
    with_std :
         (Default value = False)
    with_ndv :
         (Default value = False)
    with_count :
         (Default value = False)

    Returns
    -------

    """
    rv = f"""
    SUM({input_expression}) as sum_{as_name},
    AVG({input_expression}) as mean_{as_name},
    APPX_MEDIAN({input_expression}) as median_{as_name},
    MIN({input_expression}) as min_{as_name},
    MAX({input_expression}) as max_{as_name},"""

    if with_std:
        rv += f'\n STDDEV({input_expression}) as std_{as_name},'
    if with_ndv:
        rv += f'\n NDV({input_expression}) as unique_{as_name},'
    if with_count:
        rv += f'\n COUNT(1) as count_{as_name},'
    return rv


def get_null_count_aggr(
    columns_list, as_name='null_count', no_ending_comma=False, empty_string_null=False
):
    """Get Cloudera-valid expression counting nulls for columns.

    Parameters
    ----------
    columns_list :

    as_name :
         (Default value = 'null_count')
    no_ending_comma :
         (Default value = False)
    empty_string_null :
         (Default value = False)

    Returns
    -------

    """
    rv = ""
    pre_clause = NULL_COUNT_CLAUSE
    if empty_string_null:

        pre_clause = pre_clause.replace('IS NULL', "= ''")
    for col in columns_list:

        rv += pre_clause.format(col=col, as_col=as_name + col) + ',\n'
    if no_ending_comma:

        rv = rv.rsplit(',', 1)[0]

    return rv


def get_include_exclude_columns(cols, include_regexes=None, exclude_regexes=None):
    """Filter list by inclusion and exclusion regexes.

    Parameters
    ----------
    cols :

    include_regexes :
         (Default value = None)
    exclude_regexes :
         (Default value = None)

    Returns
    -------

    """
    if include_regexes is None:
        ret = cols
    else:
        ret = get_matching_columns(cols, include_regexes)
    ret = set(ret)
    if exclude_regexes:
        ret.difference_update(get_matching_columns(cols, exclude_regexes))
    return sorted(list(ret))


def get_matching_columns(cols: list, regex_list: list) -> list:
    """Match a list of columns with a nuber of regexes.

    Parameters
    ----------
    cols : list

    regex_list : list


    Returns
    -------


    """
    ret: List = []
    for regex in regex_list:
        regex = re.compile(regex)
        ret += filter(regex.search, cols)
    return ret


def get_sqlserver_hashed_sample_clause(id_clause, sample_pct):
    """Get SQL Server-valid synthax for hashed-sampling an id clause.on

    Takes as imput a given sample_pct in (0, 1).

    Parameters
    ----------
    id_clause :

    sample_pct :


    Returns
    -------

    """
    assert 0 < sample_pct < 1, f'{sample_pct} should be a float  in (0,1)'
    int_pct = int(sample_pct * 100)
    rv = f"""
    AND ABS(CAST(HASHBYTES('SHA1',
        {id_clause}) AS BIGINT)) % 100 <= {int_pct}"""
    return rv


def wrap_list_values_quotes(value_list: list):
    """Wraps all values in a list with single quotes

    Parameters
    ----------
    value_list: list
        Unquoted value list.

    Returns
    -------
        Single-quotes wrapped value list : list

    """
    return [f"'{value}'" for value in value_list]
