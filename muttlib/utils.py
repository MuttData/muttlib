"""Project agnostic utility functions."""
from collections import OrderedDict, deque
import contextlib
from copy import deepcopy
import csv
from datetime import date, datetime
from functools import wraps
import hashlib
import io
import logging
import logging.config
from numbers import Number
import os
from pathlib import Path
import re
import sys
from typing import List, Union, Dict, Tuple

import jinja2
from jinjasql import JinjaSql
import numpy as np
import pandas as pd
from pandas.tseries import offsets
from scipy.stats import iqr
from IPython.display import display
import matplotlib.pyplot as plt  # NOQA


logger = logging.getLogger(f'utils.{__name__}')

DEFAULT_JINJA_ENV_ARGS = dict(
    autoescape=True, line_statement_prefix="%", trim_blocks=True, lstrip_blocks=True,
)

NULL_COUNT_CLAUSE = """SUM( CASE WHEN {col} IS NULL
    THEN 1 ELSE 0 END ) AS {as_col}"""


def make_dirs(dir_path):
    """Add a return value to mkdir."""
    Path.mkdir(Path(dir_path), exist_ok=True, parents=True)
    return dir_path


def path_or_string(str_or_path):
    """Load file contents as string or return input str."""
    file_path = Path(str_or_path)
    try:
        with file_path.open('r') as f:
            return f.read()
    except (OSError, ValueError):
        return str_or_path


# PythonDecorators/decorator_function_with_arguments.py
def local_df_cache(
    # pylint: disable=unused-argument
    use_cache=False,
    refresh_cache=False,
    cache_fn='cache_file',
    cache_dir='/tmp/',  # nosec
    cache_format='pickle',
    cache_hit_msg='Reading cached data from file: ',
    cache_miss_msg='Cache missing for file: ',
    cache_put_msg='Saving data to file: ',
):
    """Decorate the target function to cache ouputed DataFrames.

    This decorator receives and removes all cache related kwargs so that the
    wrapped function doesn't get them nonetheless the caching behavior can
    be changed on the fly.
    It must be called as a function even when no arguments are passed:
    @local_df_cache()

    The decorated function must not have any params that MATCH the ones of this
    decorator.

    Parameters
    ----------
    use_cache : bool
        Whether to consult the cache or not.
    refresh_cache : bool
        Whether to update the cache or not (does not depend on use_cache).
    cache_fn : str
        Name of the cache file.
    cache_dir : str
        Directory where the cached files will be stored.
    cache_format : str
        Cache file format as supported by `df_to_multi`.
    cache_hit_msg : str
        Show this when cache hit.
    cache_miss_msg : str
        Show this when cache miss.
    cache_put_msg : str
        Show this when cache put.

    IMPORTANT: By default the filename is used as cache key,
    as such it does not consider the actual contents of the DataFrame
    being cached.
    Take care to select a cache name that reflects this changes. Such as
    including the hash of the query used to generate the data stored.

    WARNING: If the decorated function returns a tuple, only the first
    dataframe contained in the tuple will be cached.
    """
    # Scope jumping scheisse.
    orig_cache_opts = locals()

    def wrap(func):
        """Do dummy docstring."""

        @wraps(func)
        def wrapped_f(*args, **kwargs):
            """Do dummy docstring."""
            cache_opts_arg = kwargs.pop('cache_opts').copy()
            cache_opts = orig_cache_opts.copy()
            cache_opts.update(cache_opts_arg)
            use_cache = cache_opts['use_cache']
            refresh_cache = cache_opts['refresh_cache']
            cache_fn = cache_opts['cache_fn']
            cache_dir = cache_opts['cache_dir']
            cache_format = cache_opts['cache_format']
            cache_hit_msg = cache_opts['cache_hit_msg']
            cache_miss_msg = cache_opts['cache_miss_msg']
            cache_put_msg = cache_opts['cache_put_msg']

            # Clear cache args from kwargs
            for k in orig_cache_opts:
                if k in kwargs:
                    del kwargs[k]

            if callable(cache_fn):
                cache_fn = cache_fn(cache_opts, *args, **kwargs)

            base_fn = f"{cache_fn}.{cache_format}"
            cache_fn = os.path.join(cache_dir, base_fn)
            if use_cache:
                if os.access(cache_fn, os.R_OK):
                    logger.debug(f"{cache_hit_msg}{cache_fn}")
                    return df_read_multi(cache_fn)
                else:
                    logger.debug(f"{cache_miss_msg}{cache_fn}")

            rv = func(*args, **kwargs)

            if use_cache:
                if (not os.access(cache_fn, os.R_OK)) or refresh_cache:
                    logger.debug(f"{cache_put_msg}{cache_fn}")
                    # rv might be a tuple: in this case we only cache the
                    # first dataframe found in the tuple
                    if isinstance(rv, tuple):
                        for i in rv:
                            if isinstance(i, pd.DataFrame):
                                df_to_multi(i, cache_fn)
                                break
                    else:
                        df_to_multi(rv, cache_fn)

            return rv

        return wrapped_f

    return wrap


def df_read_multi(fn, index_col=False, quoting=0):
    """Read multiple table disk-formats into a pandas DataFrame."""
    ext = Path(fn).suffix[1:]
    if ext == 'csv':
        df = pd.read_csv(fn, index_col=index_col, quoting=quoting)

        def clean_quotes(s):
            """Clean start and ending quotes."""
            if s[0] in '"\'' and s[-1] in '"\'':
                return s[1:-1]
            return s

        df.columns = list(map(clean_quotes, df.columns))
        return df
    elif ext == 'feather':
        return pd.read_feather(fn)
    elif ext in ['pickle', 'pkl']:
        return pd.read_pickle(fn)
    else:
        raise ValueError(f"File format '{ext}' not supported!")


def df_to_multi(df, fn, index=False, quoting=csv.QUOTE_NONNUMERIC):
    """Convert a DF to multiple disk-formats table."""
    ext = Path(fn).suffix[1:]
    if ext == 'csv':
        return df.to_csv(fn, index=index, quoting=quoting)
    elif ext == 'feather':
        return df.to_feather(fn)
    elif ext in ['pickle', 'pkl']:
        return df.to_pickle(fn)
    else:
        raise ValueError(f"File format '{ext}' not supported!")


def convert_to_snake_case(name: str):
    """Convert string to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def deque_to_geo_hierarchy_dict(double_linked_list: deque, target_level: str):
    """Converts a deque to an ordered dictionary using GEO ordered levels."""
    orde = OrderedDict()  # type: ignore # noqa
    d = deepcopy(double_linked_list)
    while len(d) > 0:
        elem = d.popleft()
        level = elem.pop('level')
        orde[level] = elem
        if target_level == level:
            return orde


def str_to_datetime(datetime_str):
    """Convert possible date-like string to datetime object."""
    formats = (
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d',
        '%Y-%m-%d %H:%M:%S.%f',
        '%H:%M:%S.%f',
        '%H:%M:%S',
        '%Y%m%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%S',
        '%Y%m%d',
        '%Y-%m-%dT%H',
        '%Y%m',
    )
    for frmt in formats:
        try:
            return datetime.strptime(datetime_str, frmt)
        except ValueError:
            if frmt is formats[-1]:
                raise


def range_datetime(datetime_start, datetime_end, timeskip=None):
    """Build datetime generator over successive time steps."""
    if timeskip is None:
        timeskip = offsets.Day(1)
    while datetime_start <= datetime_end:
        yield datetime_start
        datetime_start += timeskip


def get_first_fortnight_last_day(ds):
    """Return the last day of the datestamp's fortnight for its month."""
    first_bday = ds + offsets.MonthBegin(1) - offsets.BMonthBegin(1)
    first_monday_second_fortnight = first_bday + offsets.BDay(10)
    last_sunday_first_fortnight = first_monday_second_fortnight - offsets.Day(1)
    return last_sunday_first_fortnight


def query_yes_no(question, default='no'):
    """Ask a yes/no question via input() and return their answer.

    Parameters
    ----------
    question : str
        Question presented to the user.
    default : str
        Default answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".

    https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    """
    valid = {'yes': True, 'y': True, 'ye': True, 'no': False, 'n': False}
    if default is None:
        prompt = ' [y/n] '
    elif default == 'yes':
        prompt = ' [Y/n] '
    elif default == 'no':
        prompt = ' [y/N] '
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()  # nosec
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def get_ordered_factor_levels(df, col, top_n=None, min_counts=None):
    """
    Return a list of a column's levels and num of levels.

    Ideallly used only on factor (categorical-typed) cols or cols without a large amount
    of values. Note that levels list are ordered by descending popularity.
    """
    rv = df[col].value_counts()
    if min_counts:
        rv = rv[rv >= min_counts]
    if top_n:
        rv = rv[:top_n]
    return rv.index.values, len(rv)


def normalize_arr(arr):
    """Normalize a numpy array to sum 1."""
    arr_sum = np.sum(arr, axis=0)
    return 1.0 * arr / arr_sum if arr_sum != 0 else arr


def apply_time_bounds(df, sd, ed, ds_col):
    """Filter time dates in a datetime-type column or index."""
    if ds_col:
        rv = df.query(f'{ds_col} >= @sd and {ds_col} <= @ed')
    else:
        rv = df.loc[sd:ed]
    return rv


def normalize_ds_index(df, ds_col):
    """Normalize usage of ds_col as column in df."""
    if ds_col in df.columns:
        return df
    elif ds_col == df.index.name:
        df = df.reset_index().rename(columns={'index': ds_col})
    else:
        raise ValueError(f"No column or index found as '{ds_col}'.")
    return df


def standarize_values(values):
    """Standarize array values with MinMAx."""
    assert np.issubdtype(values, np.number)
    shifted_values = values - values.min()
    # Degenerate case when values array has all same input values
    if np.count_nonzero(shifted_values) == 0:
        return values
    return shifted_values / (shifted_values.max() - shifted_values.min())


def robust_standarize_values(values):
    """Standarize values with InterQuartile Range and median."""
    assert np.issubdtype(values, np.number)
    return (values - values.median()) / iqr(values)


def hash_str(s, length=8):
    """Hash a string."""
    return hashlib.sha256(s.encode('utf8')).hexdigest()[:length]


def df_info_to_str(df):
    """Cast df info into string type."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()


class JinjaTemplateException(Exception):
    """Dummy doc."""


class BadInClauseException(JinjaTemplateException):
    """Dummy doc."""


def _format_value_in_clause(value: Union[Number, str]) -> str:
    """Format value according to type.

    Args:
        value (Union[Number, str]): any number or string

    Raises:
        BadInClauseException: for values other than Number or str

    Returns:
        str: formatted string
    """
    if isinstance(value, str):
        return f"'{value}'"
    elif isinstance(value, Number):
        return f"{value}"
    else:
        raise BadInClauseException(
            f"Value type: {type(value)} is not allowed for in clause formatting"
        )


def format_in_clause(
    iterable: Union[Tuple[Union[Number, str]], List[Union[Number, str]]]
) -> str:
    """
    Create a Jinja2 filter to format list-like values passed.

    Args:
        iterable (list, tuple): list / tuple of strings and numbers. Can be empty.

    Raises:
        BadInClauseException: for non iterable inputs.

    Returns:
        str: The formatted string of the list of elements.

    Notes:
        Idea originally from
        https://github.com/hashedin/jinjasql/blob/master/jinjasql/core.py
        Passing an empty tuple/list won't raise an exception, in order to
        simplify the function. Also, regarding failing queries, there's no
        explicit goal for sql formatting (although that's a common use case).

    Examples:
        >>> format_in_clause([1.12, 1, 'a'])

        (1.12,1,'a')
    """
    if not isinstance(iterable, (list, tuple)):
        raise BadInClauseException(
            f"Value passed is not a list or tuple: '{iterable}'. "
            f"Where the query uses the '| inclause'."
        )
    values = [_format_value_in_clause(v) for v in iterable]
    clause = ",".join(values)
    clause = "(" + clause + ")"
    return clause


def get_default_jinja_template(path_or_str, filters=None, **kwargs):
    """Create Jinja specific template.."""
    if filters is None:
        filters = {"inclause": format_in_clause}
    # The following line is labeled with nosec so that bandit doesn't fail. In DEFAULT_JINJA_ENV_ARGS, autoescape is set to True.
    environment = jinja2.Environment(**{**DEFAULT_JINJA_ENV_ARGS, **kwargs})  # nosec
    environment.filters = {**environment.filters, **filters}
    return environment.from_string(path_or_string(path_or_str))


def get_cloudera_sql_stats_aggr(
    input_expression,
    as_name=None,
    with_minmax=False,
    with_std=False,
    with_ndv=False,
    with_count=False,
    ends_comma=True,
):
    """Get Cloudera-valid battery of statistical aggregations clause."""
    rv_l = [
        f'SUM({input_expression}) AS sum',
        f'AVG({input_expression}) AS mean',
        f'APPX_MEDIAN({input_expression}) AS median',
    ]

    if with_minmax:
        rv_l.append(f'MIN({input_expression}) AS min')
        rv_l.append(f'MAX({input_expression}) AS max')
    if with_std:
        rv_l.append(f'STDDEV({input_expression}) AS std')
    if with_ndv:
        rv_l.append(f'NDV({input_expression}) AS unique')
    if with_count:
        rv_l.append(f'COUNT({input_expression}) AS count_rows')
    rv = ',\n'.join([f'{i}_{as_name}' for i in rv_l]) + ','
    if not ends_comma:
        rv = rv[:-1]
    return rv


def get_cloudera_sample_cut(sample_lines_ratio=None):
    """Get cut int value for sample proportion."""
    if sample_lines_ratio is None:
        sample_lines_ratio = 1.0
    # Generate the value for sample selection.
    sampling_cut = int(2 ** 64 * sample_lines_ratio / 2.0) - 1
    return sampling_cut


def get_cloudera_hashed_sample_clause(col_or_exp, sample_pct):
    """
    Get Cloudera-valid clause for hashed-sampling.

    This will work on an id col or on a given expression that outputs
    a valid column. It takes a sample_pct number between 0 and 1.
    """
    assert 0 < sample_pct < 1, f'{sample_pct} should be a float  in (0,1)'

    threshold_int = get_cloudera_sample_cut(sample_lines_ratio=sample_pct)
    rv = f'AND abs(fnv_hash(CAST({col_or_exp} AS bigint))) <= {threshold_int}'

    return rv


def str_normalize_pandas(data, str_replace_kws=None):
    """Normalize all string-like data in pandas objects.

    Parameters
    ----------
    data (pd.DataFrame, pd.Series): containing the data. Might or not have string
        columns
    str_replace_kws (dict): contains pandas str.replace method kwargs
    """

    if isinstance(data, pd.DataFrame):
        obj_cols = data.select_dtypes(include=[np.object]).columns
        for col in obj_cols:
            data[col] = (
                data[col]
                .str.lower()
                .str.normalize('NFKD')
                .str.encode('ascii', errors='ignore')
                .str.decode('utf-8')
            )
            if str_replace_kws:
                data[col] = data[col].str.replace(**str_replace_kws)
        return data
    elif isinstance(data, pd.Series) and data.dtype == np.object:
        data = (
            data.str.lower()
            .str.lower()
            .str.normalize('NFKD')
            .str.encode('ascii', errors='ignore')
            .str.decode('utf-8')
        )
        if str_replace_kws:
            data = data.str.replace(**str_replace_kws)
        return data
    else:
        raise TypeError(f"File format '{type(data)}' not supported!")


def df_optimize_float_types(
    df, type_mappings: Dict[str, str] = None,
):
    """Cast dataframe columns to more memory friendly types.

    Parameters
    ----------
        df: DataFrame to be modified.
        type_mappings: Mapping of types. Defaults to {"float64":"float16", "float32":"float16"}

    WARNING: Type conversion leads to a loss in accuracy and possible overflow of the target type.
    Eg:
    >>> n = 2**128
    >>> np.float64(n), np.float32(n)
    (3.402823669209385e+38, inf)
    """
    if type_mappings is None:
        type_mappings = {
            "float64": "float16",
            "float32": "float16",
        }

    new_dtypes = {c: type_mappings.get(t.name, t) for c, t in df.dtypes.iteritems()}
    df = df.astype(new_dtypes, copy=False)
    return df


def df_replace_empty_strs_null(df):
    """Replace whitespace or empty strs with nan values."""
    str_cols = df.select_dtypes(include='object').columns.tolist()
    if str_cols:
        logger.debug(f'Replacing whitespace in these object cols: {str_cols}...')
        for col in str_cols:
            df[col].replace(r'^\s*$', np.nan, regex=True, inplace=True)
    return df


def df_drop_nulls(df, max_null_prop=0.2, protected_cols: List[str] = None):
    """Drop null columns in df, for null share over a certain threshold."""
    # Note: Pandas treats string columns as `object` data types.
    # Warning this function modifies the passed df. If you dont want this you should use df.copy()
    if protected_cols is None:
        protected_cols = list()
    logger.debug(
        f'Dropping columns with null ratio greater than {max_null_prop:.2%}...'
    )
    df = df_replace_empty_strs_null(df)
    null_means = df.isnull().mean()
    null_mask = null_means < max_null_prop

    null_mask[[c for c in df.columns if c in protected_cols]] = True
    drop_cols = null_mask[~null_mask].index.tolist()

    logger.debug(
        f'Null proportions:\n'
        f'{null_means.loc[drop_cols].sort_values(ascending=False)}'
    )

    logger.debug(f'Dropping the following {len(drop_cols)} columns:\n {drop_cols}')
    df.drop(drop_cols, axis=1, inplace=True)

    return df


def df_drop_std(df, min_std_dev=1.5e-2, protected_cols: List[str] = None):
    """Drop low variance cols."""
    # Warning this function modifies the passed df. If you dont want this you should use df.copy()
    if protected_cols is None:
        protected_cols = list()
    std_values = df.std()
    low_variance_cols = std_values < min_std_dev
    low_variance_cols = low_variance_cols.index[low_variance_cols].tolist()
    low_variance_cols = [c for c in low_variance_cols if c not in protected_cols]
    logger.debug(
        f'Dropping the following {len(low_variance_cols)} columns '
        f'due to low variance:\n {low_variance_cols}'
    )
    df.drop(low_variance_cols, axis=1, inplace=True)
    return df


def df_drop_corr(
    df,
    target_col,
    max_corr=0.3,
    protected_cols: List[str] = None,
    frac=0.2,
    random_state=None,
):
    """Drop high correlated to-target cols."""
    # Warning this function modifies the passed df. If you dont want this you should use df.copy()

    if target_col not in df.columns:
        raise ValueError(f"target col ({target_col}) is not in dataframe columns")

    if protected_cols is None:
        protected_cols = list()

    corr_df = df.sample(frac=frac, random_state=random_state).corr()
    high_corr_cols = abs(corr_df[target_col]) > max_corr
    high_corr_cols = high_corr_cols.index[high_corr_cols].tolist()
    high_corr_cols = [c for c in high_corr_cols if c not in protected_cols]
    logger.debug(
        f'Dropping the following {len(high_corr_cols)} columns due to high correlation '
        f'with target:\n {high_corr_cols}'
    )
    df.drop(high_corr_cols, axis=1, inplace=True)
    return df


def df_get_typed_cols(df, col_type='cat', protected_cols: List[str] = None):
    """Get typed columns, excluding protected cols if passed."""
    assert col_type in ('cat', 'num', 'date', 'bool', 'timedelta')
    if protected_cols is None:
        protected_cols = list()

    if col_type == 'cat':  # Work in cases, else dont define include var
        include = ['object', 'category']
    elif col_type == 'num':
        include = ['number']
    elif col_type == 'date':
        include = ['datetime']
    elif col_type in ('bool', 'timedelta'):
        include = [col_type]
    typed_cols = [
        c for c in df.select_dtypes(include=include).columns if c not in protected_cols
    ]
    return typed_cols


def df_encode_categorical_dummies(
    df,
    cat_cols: List[str] = None,
    skip_cols: List[str] = None,
    top=25,
    other_val='OTHER',
):
    """Encode categorical columns into dummies."""
    if skip_cols is None:
        skip_cols = list()
    if cat_cols is None:
        cat_cols = list()
    pre_dummy_cols = df.columns.tolist()
    cat_cols = df_get_typed_cols(df, col_type='cat') if cat_cols == [] else cat_cols
    cat_cols = [c for c in cat_cols if c not in skip_cols]

    for c in cat_cols:
        top_categories = df[c].value_counts().index.values[0:top]
        df[c] = df[c].where(df[c].isin(top_categories), other=other_val)

    logger.debug(f'Getting dummies from these top categories:{cat_cols}...')
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    dummy_cols = list(set(df.columns.tolist()) - set(pre_dummy_cols))
    logger.debug(
        f'{len(df.columns)} columns after dummies:\n {sorted(df.columns.tolist())}'
    )
    return df, dummy_cols


def df_drop_single_factor_level(df):
    """Drop categorical columns with null or 1 level."""
    # Warning this function modifies the passed df. If you dont want this you should use df.copy()
    cat_cols = df_get_typed_cols(df, col_type='cat')
    cols_to_drop = []
    for c in cat_cols:
        val_count = df[c].value_counts(dropna=False)
        vals = val_count.index.tolist()
        if (len(vals) == 2 and '' in vals) or (len(vals) == 1):
            cols_to_drop.append(c)
    logger.debug(
        f'Dropping the following {len(cols_to_drop)} columns with low factor levels:'
        f'\n {cols_to_drop}.'
    )
    df.drop(cols_to_drop, axis=1, inplace=True)
    return df


def dedup_list(li: list):
    """
    Deduplicates list while preserving order.

    Note:
        Not the same as `list(set(li))`.
        Copied from https://stackoverflow.com/questions/31479188/quickest-way-to-dedupe-list-in-dict

    Args:
        li (list): list of elements. Can be empty.

    Returns:
        list: The deduplicated list.

    Examples:
        >>> dedup_list([1, 1, 'a'])
        [1, 'a']

        >>> dedup_list([])
        []
    """
    assert isinstance(
        li, list
    ), f"Input argument should be a list. Value passed was: {li}."
    new_list: List[Union[int, str, list]] = []
    for val in li:
        if val not in new_list:
            new_list.append(val)
    return new_list


def create_forecaster_dates(end_date, forecast_train_window, forecast_future_window):
    """Process and correct all respective dates for forecaster.

    Args:
        end_date (datetime): The end date with hour precision.
        forecast_train_window (int): The window days for past data.
        forecast_future_window (int): The window days for future predictions.

    Returns:
        (datetime, datetime, datetime): Truple with start, end and future dates.
    """
    if not all([forecast_future_window > 0, forecast_train_window >= 0]):
        raise ValueError(
            f"Future ('{forecast_future_window}') or train "
            f"('{forecast_train_window}') windows are not geq 0."
        )

    if not (
        isinstance(end_date, str)
        or isinstance(end_date, date)
        or isinstance(end_date, pd.Timestamp)
    ):
        raise TypeError("`end_date` arg must be date-like or string typed")

    end_date = str_to_datetime(end_date) if isinstance(end_date, str) else end_date
    start_date = end_date - pd.offsets.Day(forecast_train_window)
    future_date = end_date + pd.offsets.Day(forecast_future_window)
    return start_date, end_date, future_date


def get_matching_columns(cols, regex_list):
    """Match a list of columns with a number of regexes."""
    ret = []
    for regex in regex_list:
        regex = re.compile(regex)
        ret += filter(regex.search, cols)
    return ret


def get_include_exclude_columns(cols, include_regexes=None, exclude_regexes=None):
    """Filter list by inclusion and exclusion regexes."""
    if not cols:
        raise ValueError("`cols` argument must be a non-empty list")
    if include_regexes is None:
        ret = cols
    else:
        ret = get_matching_columns(cols, include_regexes)
    ret = set(ret)
    if exclude_regexes:
        ret.difference_update(get_matching_columns(cols, exclude_regexes))

    return sorted(list(ret))


def dataframe_diff(df_x, df_y, key, right_suffix="_x", left_suffix="_y"):
    """Compute diff between 2 dataframes taking their columns in common as key.

    Parameters
    ---------
    df_x : DataFrame
        First df.
    df_y : DataFrame
        Second df.
    key : list of str
        List of column names taken as keys.
    right_suffix : str
        Suffix for the right side dataframe.
    left_suffix : str
        Suffix for the elft side dataframe.

    Returns
    -------
    tuple of DataFrame
        Returns a tuple that contains the difference and additional data.

        First df contains the following columns:
        * key: list of columns taken as group columns
        * value_x: different value of x side
        * value_y: different value of y side
        * column_name: contains the name of column where the change is produced

        Second df contains the following columns:
        * key: list of columns taken as group columns
        * other columns in common between df_x and df_y that contain additional changes
        * sets: contains the name of df where the change is produced (df_x or df_y)

    Refs
    * https://github.com/yogiadi/dataframe_diff/blob/master/dataframe_diff/dataframe_diff.py
    """
    set_x = [f"df{right_suffix}" for i in range(len(df_x))]
    df_x['sets'] = set_x
    set_y = [f"df{left_suffix}" for i in range(len(df_y))]
    df_y['sets'] = set_y
    columns = list(df_x.columns)
    columns.remove('sets')
    df_concat = (
        pd.concat([df_x, df_y])
        .drop_duplicates(subset=columns, keep=False)
        .reset_index(drop=True)
    )
    df_set1 = df_concat[df_concat['sets'] == f"df{right_suffix}"]
    df_set2 = df_concat[df_concat['sets'] == f"df{left_suffix}"]
    df_merged = pd.merge(df_set1, df_set2, on=key)
    nonkey = set(columns) - set(key)
    list_diff = []
    for i in range(len(df_merged)):
        for col in nonkey:
            if (
                df_merged.iloc[i][col + right_suffix]
                != df_merged.iloc[i][col + left_suffix]
            ):
                list_diff.append(
                    list(
                        df_merged.iloc[i][key + [col + right_suffix, col + left_suffix]]
                    )
                    + [col]
                )
    df_diff = pd.DataFrame(
        list_diff,
        columns=key + ['value' + right_suffix, 'value' + left_suffix, 'column_name'],
    )
    df_additional = (
        pd.concat([df_x, df_y])
        .drop_duplicates(subset=key, keep=False)
        .reset_index(drop=True)
    )
    df_x.drop(['sets'], axis=1, inplace=True)
    df_y.drop(['sets'], axis=1, inplace=True)
    return df_diff, df_additional


def compute_differences_dataframes(
    first_df,
    second_df,
    key_cols,
    first_suffix,
    second_suffix,
    filter_flag_more_deviation=False,
    threshold=1,
):
    """
    It generates the differentials between dataframes.

    Args:
        first_df (pandas.DataFrame): First dataframe to be taken to compute differences.
        second_df (pandas.DataFrame): Second dataframe to be taken to compute differences.
        key_cols (list of strings): list of column names to take as key (to join and sort)
        first_suffix (string): suffix to name row_count column
        second_suffix (string): suffix to name row_count column
        filter_flag_more_deviation (bool): Flag that indicates if transform function should filter or not data
        if the data has deviation greater than specified value (threshold)
        threshold (float): value that determines filtering of data.

    Returns:
        DataFrame: Result df with deviations. This df contains the followings columns as result:
            - params["key_col"] (list of string/int): list of column names to take as key.
            - row_count{params['first_suffix']} (int): row count of first df
            - row_count{params['second_suffix']} (int): row count of second df
            - diff (int): difference between row_count of first df and row_count of second df
            - diff_% (float): diff in percentage units
    """

    df_merged = first_df.merge(
        second_df, how="left", on=key_cols, suffixes=(first_suffix, second_suffix)
    ).fillna(value=0)
    df_merged["diff"] = df_merged[f"row_count{first_suffix}"] - (
        df_merged[f"row_count{second_suffix}"]
    )
    df_merged["diff_%"] = round(
        (
            (
                df_merged[f"row_count{first_suffix}"]
                - (df_merged[f"row_count{second_suffix}"])
            )
            / df_merged[f"row_count{first_suffix}"]
        )
        * 100,
        2,
    )

    if filter_flag_more_deviation:
        df_merged = df_merged[
            ~df_merged["diff_%"].between(-threshold, threshold, inclusive=False)
        ]
    df_merged = df_merged[
        key_cols
        + [f"row_count{first_suffix}", f"row_count{second_suffix}", "diff", "diff_%"]
    ]

    return df_merged.sort_values(key_cols, ascending=True).reset_index(drop=True)


@contextlib.contextmanager
def numpy_temp_seed(seed=42):
    """Sets numpy temporary random state globally.

    Args:
        seed (int): Seed for RandomState.

    Yields:
        None

    Raises:
        TypeError: if `seed` argument isn't an instance of `int`.

    Examples:

        >>> with numpy_temp_seed(333):
        ...     numpy.random.randint(10)
        3

        >>> with numpy_temp_seed(388681):
        ...     some_function_that_generates_numpy_random_numbers()

    Notes:
        Must be used within a `with` context.
    """
    if not isinstance(seed, int):
        raise TypeError(f"`seed` argument must be int typed, not {type(seed)}")
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        yield
    finally:
        np.random.set_state(state)


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
    test_id_digest = hashlib.md5(test_id.encode('ascii')).hexdigest()  # nosec
    test_id_first_digits = test_id_digest[:6]
    test_id_last_int = int(test_id_first_digits, 16)
    split = test_id_last_int / 0xFFFFFF
    return split > control_group_size


def col_sample_display(
    df: pd.DataFrame,
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
    if col in df_get_typed_cols(df, 'date'):
        # Direct exit with `date`, as they can be miscasted to numeric
        is_numeric_type = False
    else:
        if col in df_get_typed_cols(df, 'num'):
            is_numeric_type = True
        else:  # cols that are string which might get converted
            try:
                pd.to_numeric(df[col].sample(num_sample))
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
        num_col[col] = pd.to_numeric(num_col[col].values, errors='coerce')
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


def sum_count_aggregation(
    df: pd.DataFrame,
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
    df: pd.DataFrame,
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
    top_levels, _ = get_ordered_factor_levels(df, categorical_col, n_levels - 1)

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
        lambda x: sub_categorize(x, top_levels) if (pd.notnull(x)) else x
    )

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
        missing_placeholders = [
            k for k, v in bind_params.items() if jinja2.Undefined() == v
        ]

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
