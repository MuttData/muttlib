"""Project agnostic utility functions."""
import csv
import hashlib
import io
import logging
import logging.config
import os
import re
import sys

from collections import OrderedDict, deque, namedtuple
from copy import deepcopy
from datetime import datetime
from functools import wraps
from pathlib import Path

import jinja2
import pandas as pd
import yaml
from pandas.tseries import offsets
from scipy.stats import iqr

logger = logging.getLogger(f'utils.{__name__}')


def read_yaml(f):
    """Read a yaml file."""
    with open(f, 'r') as file:
        return yaml.safe_load(file.read())


def make_dirs(dir_path):
    """Add a return value to mkdir."""
    Path.mkdir(Path(dir_path), exist_ok=True, parents=True)
    return dir_path


def non_empty_dirs(path):
    """List all non-empty directories for a given path."""
    return list({str(p.parent) for p in path.rglob('*') if p.is_file()})


def path_or_string(str_or_path):
    """Load file contents as string or return input str."""
    file_path = Path(str_or_path)
    try:
        if file_path.is_file():
            with file_path.open('r') as f:
                return f.read()
    except OSError:
        pass
    return str_or_path


# PythonDecorators/decorator_function_with_arguments.py
def local_df_cache(
    # pylint: disable=unused-argument
    use_cache=False,
    refresh_cache=False,
    cache_fn='cache_file',
    cache_dir='/tmp/',
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

    The decorated function must not have any params that match the ones of this
    decorator.

    Args:
        use_cache: Whether to consult the cache or not.
        refresh_cache: Whether to update the cache or not (does not depend on
                       use_cache).
        cache_fn: Name of the cache file.
        cache_dir: Directory where the cached files will be stored.
        cache_format: Cache file format.
        cache_hit_msg: Show this when cache hit.
        cache_miss_msg: Show this when cache miss.
        cache_put_msg: Show this when cache put.

    Cache format is defined by the ones that df_to_multi and
    df_to_multi support.

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
            else:
                cache_fn = cache_fn

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


def dict_to_namedtuple(name, d):
    """Convert dictionary to namedtuple object."""
    return namedtuple(name, d.keys())(**d)


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


def wrap_list_values_quotes(lis):
    """Wraps all values in a list with single quotes."""
    return [f"'{val}'" for val in lis]


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
    for format in formats:
        try:
            return datetime.strptime(datetime_str, format)
        except ValueError:
            if format is formats[-1]:
                raise


def range_datetime(datetime_start, datetime_end, timeskip=None):
    """Build datetime generator over successive time steps."""
    if timeskip is None:
        timeskip = offsets.Day(1)
    while datetime_start <= datetime_end:
        yield datetime_start
        datetime_start += timeskip


def get_fathers_mothers_kids_day(year: int):
    """Get three dates for a given year input."""
    august_first = str_to_datetime(f'{year}-08-01')
    kids_dow_iteration = 3  # third sunday each August
    KIDS_DAY = pd.date_range(
        start=august_first, end=august_first + offsets.Day(21), freq='W-SUN'
    )[kids_dow_iteration - 1]

    october_first = august_first + offsets.MonthBegin(2)
    mother_dow_iteration = 3  # third sunday each October
    MOTHERS_DAY = pd.date_range(
        start=october_first, end=october_first + offsets.Day(21), freq='W-SUN'
    )[mother_dow_iteration - 1]

    june_first = august_first - offsets.MonthBegin(2)
    father_dow_iteration = 3  # third sunday each June
    FATHERS_DAY = pd.date_range(
        start=june_first, end=june_first + offsets.Day(21), freq='W-SUN'
    )[father_dow_iteration - 1]

    return FATHERS_DAY, MOTHERS_DAY, KIDS_DAY


def get_friends_day(year: int):
    """Get arg-style friends date."""
    return str_to_datetime(f'{year}-07-20')


def is_special_day(ds, timestamps_inclause):
    """
    Flag a date or date-string-like object as a special day.

    The default list of timestamps will be the `Celebration days` which are
    those commercially imposed days (not easter, EOY, christmas, etc.) that are
    not technically `Feriados` per se.
    """
    if isinstance(ds, str):
        ds = str_to_datetime(ds).date()
    else:
        ds = ds.date()
    # breakpoint()
    dates_inclause = [d.date() for d in timestamps_inclause]
    if ds in dates_inclause:
        return 1
    else:
        return 0


def get_semi_month_pay_days(start_date, end_date):
    """Get half and end of month friday pay-days for salary workers."""
    first_monthly_fridays = pd.date_range(
        start=start_date, end=end_date, freq='BM'
    ) + offsets.Week(
        weekday=4
    )  # move to end of month
    # Beware fridays that are too faraway from middle/end of month
    first_semimonth_pay_days = [
        ts + offsets.Day(14) if ts.day <= 4 else ts + offsets.Day(7)
        for ts in first_monthly_fridays
    ]
    second_semimonth_pay_days = [
        ts + offsets.Day(14) for ts in first_semimonth_pay_days
    ]
    SEMIMONTH_PAY_DAYS = sorted(first_semimonth_pay_days + second_semimonth_pay_days)
    return SEMIMONTH_PAY_DAYS


def get_first_fortnight_last_day(ds):
    """Return the last day of the datestamp's fortnight for its month."""
    first_bday = ds + offsets.MonthBegin(1) - offsets.BMonthBegin(1)
    first_monday_second_fortnight = first_bday + offsets.BDay(10)
    last_sunday_first_fortnight = first_monday_second_fortnight - offsets.Day(1)
    return last_sunday_first_fortnight


def create_dict_id(d, length=10):
    """Return SHAKE 256 hash from dict.

    Args:
        d: dict containing all necesssary data
        length: number of characters to return in the hash
    """
    shake = hashlib.shake_256()
    shake.update(repr(d).encode('utf-8'))
    return shake.hexdigest(int(length / 2))


def query_yes_no(question, default='no'):
    """Ask a yes/no question via input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
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
        choice = input().lower()
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
    arr_sum = pd.np.sum(arr, axis=0)
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
    shifted_values = values - values.min()
    # Degenerate case when values array has all same input values
    if pd.np.count_nonzero(shifted_values) == 0:
        return values
    return shifted_values / (shifted_values.max() - shifted_values.min())


def robust_standarize_values(values):
    """Standarize values with InterQuartile Range and median."""
    return (values - values.median()) / iqr(values)


def none_or_empty_pandas(obj):
    """Check if object is None or empty pd.Dataframe / pd.Series."""
    if obj is None:
        return True
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.empty
    else:
        raise ValueError(
            "Argument type should be one of: "
            f"'{(type(None) , pd.Series, pd.DataFrame)}'. Type passed was {type(obj)}"
        )


def hash_str(s, length=8):
    """Hash a string."""
    return hashlib.sha256(s.encode('utf8')).hexdigest()[:length]


def setup_logging(log_config, logger_name='root', level='INFO'):
    """Setup logging config."""
    log_config['loggers'][logger_name]['level'] = level
    logging.config.dictConfig(log_config)


def df_info_to_str(df):
    """Cast df info into string type."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()


class JinjaTemplateException(Exception):
    """Dummy doc."""

    pass


class BadInClauseException(JinjaTemplateException):
    """Dummy doc."""

    pass


def in_clause_requirement(obj):
    """Check object list or tuple iterables."""
    return isinstance(obj, (list, tuple))


def format_in_clause(iterable):
    """
    Create a Jinja2 filter to format list-like values passed.

    Idea originally from
    https://github.com/hashedin/jinjasql/blob/master/jinjasql/core.py
    """
    if not in_clause_requirement(iterable):
        raise BadInClauseException(
            f"Value passed is not a list or tuple: '{iterable}'. "
            f"Where the query uses the '| inclause'."
        )
    values = [f"{v}" for v in iterable]
    clause = ",".join(values)
    clause = "(" + clause + ")"
    return clause


def template(path_or_str, **kwargs):
    """Create jinja specific template.."""
    environment = jinja2.Environment(
        line_statement_prefix=kwargs.pop('line_statement_prefix', '%'),
        trim_blocks=kwargs.pop('trim_blocks', True),
        lstrip_blocks=kwargs.pop('lstrip_blocks', True),
        **kwargs,
    )
    environment.filters['inclause'] = format_in_clause
    return environment.from_string(path_or_string(path_or_str))


def render_jinja_template(path_or_str, jparams={}):
    """Render a query with jinja, from a complete path to a sql-like file."""
    # Standarize to pathlib object, supports str objects
    pat = Path(path_or_str).expanduser().resolve().as_posix()
    if pd.np.DataSource().exists(pat):
        logger.debug(f'Loading jinja template from {pat}.')
    return template(path_or_str).render(**jparams)


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
        obj_cols = data.select_dtypes(include=[pd.np.object]).columns
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

    elif isinstance(data, pd.Series) and data.dtype == pd.np.object:
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


def df_optimize_float_types(
    df, type_mappings={"float64": "float16", "float32": "float16"}
):
    """Cast dataframe columns to more memory friendly types.

    WARNING: Type conversion leads to a loss in accuracy and possible overflow of the target type.
    Eg:
    >>> n = 2**128
    >>> np.float64(n), np.float32(n)
    (3.402823669209385e+38, inf)
    """
    new_dtypes = {c: type_mappings.get(t.name, t) for c, t in df.dtypes.iteritems()}
    df = df.astype(new_dtypes, copy=False)
    return df


def df_replace_empty_strs_null(df):
    """Replace whitespace or empty strs with nan values."""
    str_cols = df.select_dtypes(include='object').columns.tolist()
    if str_cols:
        logger.debug(f'Replacing whitespace in these object cols: {str_cols}...')
        df[str_cols].replace(r'^\s*$', pd.np.nan, regex=True, inplace=True)
    return df


def df_drop_nulls(df, max_null_prop=0.2, protected_cols=[]):
    """Drop null columns in df, for null share over a certain threshold."""
    # Note: Pandas treats string columns as `object` data types
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


def df_drop_std(df, min_std_dev=1.5e-2, protected_cols=[]):
    """Drop low variance cols."""
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


def df_drop_corr(df, target_col, max_corr=0.3, protected_cols=[]):
    """Drop high correlated to-target cols."""
    assert target_col in df.columns
    corr_df = df.sample(frac=0.2).corr()
    high_corr_cols = (abs(corr_df[target_col]) > max_corr)
    high_corr_cols = high_corr_cols.index[high_corr_cols].tolist()
    high_corr_cols = [
        c for c in high_corr_cols if c not in protected_cols]
    logger.debug(
        f'Dropping the following {len(high_corr_cols)} columns due to high correlation '
        f'with target:\n {high_corr_cols}'
    )
    df.drop(high_corr_cols, axis=1, inplace=True)
    return df


def df_get_typed_cols(df, col_type='cat', protected_cols=[]):
    """Get typed columns, excluding protected cols if passed."""
    assert col_type in ('cat', 'num', 'date', 'bool', 'timedelta')
    if col_type == 'cat':  # Work in cases, else dont define include var
        include = ['object', 'category']
    elif col_type == 'num':
        include = [pd.np.number]
    elif col_type == 'date':
        include = ['datetime']
    elif col_type in ('bool', 'timedelta'):
        include = [col_type]
    typed_cols = [
        c for c in df.select_dtypes(include=include).columns if c not in protected_cols
    ]
    return typed_cols


def df_encode_categorical_dummies(
    df, cat_cols=[], skip_cols=[], top=25, other_val='OTHER'
):
    """Encode categorical columns into dummies."""
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
