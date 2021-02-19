# pylint:disable=W0611, E1101

from collections import OrderedDict, deque, namedtuple  # noqa: F401
import datetime
from pathlib import Path
from textwrap import dedent
import io

import numpy as np
import pandas as pd
from pandas._testing import assert_frame_equal
from pandas.tseries import offsets
import pytest
from unittest import mock

from muttlib import utils

## MOCKS FOR DIFF DATAFRAME AND DEVIATION_LOG


def get_first_df_diff_deviations_functions(params=None):
    data1 = {
        "date_col": ["2020-11-11", "2020-11-12", "2020-11-13", "2020-11-14"],
        "row_count": [20, 100, 10, 1],
    }
    return pd.DataFrame(data1)


def get_second_df_diff_deviations_functions(params=None):
    data2 = {
        "date_col": ["2020-11-11", "2020-11-12", "2020-11-13", "2020-11-14"],
        "row_count": [10, 20, 20, 10],
    }
    return pd.DataFrame(data2)


# ------
@pytest.fixture()
def sample_df():
    base_date = np.datetime64('2020-01-01')
    n_rows = 100 ** 2
    with utils.numpy_temp_seed():
        df = pd.DataFrame(
            {
                'id': np.arange(n_rows),
                'batman': np.random.randint(1, 5, n_rows),
                'robin': np.random.randint(1, 1000, n_rows),
            }
        )
        df['str_robin'] = df.robin.astype(str)
        df['riddler'] = df.batman.replace([1, 2, 3, 4], ['a', 'b', 'c', 'd'])
        df['two_face'] = df.batman.apply(lambda x: base_date + np.random.choice(x))
    return df


@pytest.fixture()
def sample_timeseries_df():
    return pd.DataFrame(
        {
            'two_face': {
                0: pd.Timestamp('2020-01-01 00:00:00'),
                1: pd.Timestamp('2020-01-02 00:00:00'),
                2: pd.Timestamp('2020-01-03 00:00:00'),
                3: pd.Timestamp('2020-01-04 00:00:00'),
            },
            'batman_sum': {0: 9812, 1: 7703, 2: 5024, 3: 2404},
            'batman_count': {0: 5173, 1: 2764, 2: 1462, 3: 601},
            'robin_sum': {0: 2617661, 1: 1374750, 2: 726399, 3: 297004},
            'robin_count': {0: 5173, 1: 2764, 2: 1462, 3: 601},
        }
    )


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ('2019-10-25 18:35:22', datetime.datetime(2019, 10, 25, 18, 35, 22)),
        ('2019-10-25', datetime.datetime(2019, 10, 25, 0, 0)),
        (
            '2019-10-25 18:35:22.000333',
            datetime.datetime(2019, 10, 25, 18, 35, 22, 333),
        ),
        ('18:35:22.000333', datetime.datetime(1900, 1, 1, 18, 35, 22, 333)),
        ('18:35:22', datetime.datetime(1900, 1, 1, 18, 35, 22)),
        ('20191025T18:35:22', datetime.datetime(2019, 10, 25, 18, 35, 22)),
        ('2019-10-25T18:35:22', datetime.datetime(2019, 10, 25, 18, 35, 22)),
        ('20191025', datetime.datetime(2019, 10, 25, 0, 0)),
        ('2019-10-25T18', datetime.datetime(2019, 10, 25, 18, 0)),
        ('201910', datetime.datetime(2019, 10, 1, 0, 0)),
    ],
)
def test_str_to_datetime(test_input, expected):
    # Testing all posible datetime formats and the exception for wrong format
    assert utils.str_to_datetime(test_input) == expected
    with pytest.raises(ValueError):
        utils.str_to_datetime("25/10/2019")


def test_get_ordered_factor_levels():
    lst = ['cat', 'dog', 'cat', 'dog', 'cat', 'dog', 'horse', 'dog', 'horse']
    df = pd.DataFrame({'olmcdonald': lst}, columns=['olmcdonald'])
    factor_levels = utils.get_ordered_factor_levels(df, 'olmcdonald')

    assert np.array_equal(np.array(['dog', 'cat', 'horse']), factor_levels[0])
    assert 3 == factor_levels[1]
    # Testing the other params
    factor_levels = utils.get_ordered_factor_levels(df, 'olmcdonald', 2)

    assert np.array_equal(np.array(['dog', 'cat']), factor_levels[0])
    assert 2 == factor_levels[1]

    factor_levels = utils.get_ordered_factor_levels(df, 'olmcdonald', 2, 4)

    assert np.array_equal(np.array(['dog']), factor_levels[0])
    assert 1 == factor_levels[1]


def test_query_yes_no(monkeypatch):
    # Testing the possible defaults and rewriting the input for the valid or invalid inputs
    og = utils.__builtins__["input"]
    with pytest.raises(ValueError):
        utils.query_yes_no("hit or miss?", 's')
    monkeypatch.setattr('builtins.input', lambda: 'yes')
    assert utils.query_yes_no("hit or miss?", default=None)
    monkeypatch.setattr('builtins.input', lambda: '')
    assert not utils.query_yes_no("hit or miss?")
    monkeypatch.setattr('builtins.input', lambda: '')
    assert utils.query_yes_no("hit or miss?", 'yes')
    monkeypatch.setattr('builtins.input', lambda: 'yes')
    assert utils.query_yes_no("hit or miss?")
    monkeypatch.setattr('builtins.input', lambda: 'y')
    assert utils.query_yes_no("hit or miss?")
    monkeypatch.setattr('builtins.input', lambda: 'ye')
    assert utils.query_yes_no("hit or miss?")
    monkeypatch.setattr('builtins.input', lambda: 'no')
    assert not utils.query_yes_no("hit or miss?")
    monkeypatch.setattr('builtins.input', lambda: 'n')
    assert not utils.query_yes_no("hit or miss?")
    monkeypatch.setattr('builtins.input', ['y', 'miss', 'miss'].pop)
    assert utils.query_yes_no("hit or miss?")

    utils.__builtins__["input"] = og


def test_path_or_string(tmpdir):
    # generate a tmp file for this test
    p = tmpdir.mkdir("sub").join("test.txt")
    p.write("True")
    assert 'True' == utils.path_or_string(p)
    assert 'show me what you got' == utils.path_or_string("show me what you got")


def test_hash_str():
    assert '0c5024ed' == utils.hash_str("hit or miss")


def test_deque_to_geo_hierarchy_dict():
    # Testing the creation of the orderedDict for the 4 levels
    lst = [
        {'level': 'National', 'select_clause': '', 'group_clause': ''},
        {
            'level': 'Provincial',
            'select_clause': "existence is pain",
            'post_join_select': 'province_name,',
            'group_clause': '1,',
        },
        {
            'level': 'Departamental',
            'select_clause': "existence is pain",
            'post_join_select': 'departament_name,',
            'group_clause': '2,',
        },
        {
            'level': 'Local',
            'select_clause': "existence is pain",
            'post_join_select': 'locality_name,',
            'group_clause': '3,',
        },
    ]
    deque_lst = deque(lst)
    cont_lst = [
        ('National', {'select_clause': '', 'group_clause': ''}),
        (
            'Provincial',
            {
                'select_clause': "existence is pain",
                'post_join_select': 'province_name,',
                'group_clause': '1,',
            },
        ),
        (
            'Departamental',
            {
                'select_clause': "existence is pain",
                'post_join_select': 'departament_name,',
                'group_clause': '2,',
            },
        ),
        (
            'Local',
            {
                'select_clause': "existence is pain",
                'post_join_select': 'locality_name,',
                'group_clause': '3,',
            },
        ),
    ]

    assert OrderedDict(cont_lst[:1]) == utils.deque_to_geo_hierarchy_dict(
        deque_lst, 'National'
    )
    assert OrderedDict(cont_lst[:2]) == utils.deque_to_geo_hierarchy_dict(
        deque_lst, 'Provincial'
    )
    assert OrderedDict(cont_lst[:3]) == utils.deque_to_geo_hierarchy_dict(
        deque_lst, 'Departamental'
    )
    assert OrderedDict(cont_lst[:]) == utils.deque_to_geo_hierarchy_dict(
        deque_lst, 'Local'
    )


def test_df_info_to_str():
    # Almost the same `test_info_memory`:
    # https://github.com/pandas-dev/pandas/blob/4edf938aedf55b9e6fbfb3199f70f857e8ec7e41/pandas/tests/frame/test_repr_info.py#L209
    df = pd.DataFrame({'B': [25, 94, 57, 62, 70]}, columns=['B'])
    result = utils.df_info_to_str(df)
    byt = float(df.memory_usage().sum())
    # Manually imput the df's `memory-size` value, as they are platform dependent i.e.
    # different platforms allocate different memory sizes for, say, int64 types
    expected = dedent(
        f"""\
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5 entries, 0 to 4
    Data columns (total 1 columns):
     #   Column  Non-Null Count  Dtype
    ---  ------  --------------  -----
     0   B       5 non-null      int64
    dtypes: int64(1)
    memory usage: {byt} bytes
    """
    )
    assert result == expected


def test_get_default_jinja_template_basic(tmpdir):
    tmp_template = (
        "Hello,my name is {{name}} you kill my {{daddy}} prepare to {{acction}}"
    )
    str_template = "Hello,my name is Inigo Montoya you kill my father prepare to die"

    p = tmpdir.mkdir("sub").join("template_test.html")
    p.write(tmp_template)

    assert str_template == utils.get_default_jinja_template(p).render(
        name="Inigo Montoya", daddy="father", acction="die"
    )


def test_get_default_jinja_template_macros(tmpdir):
    p = tmpdir.mkdir("sub").join("macros_test.html")
    # testing macros
    macro = "{% macro test_macro(name) %}" "Hello lil {{ name }}" "{% endmacro %}"
    p.write(macro)
    out = utils.get_default_jinja_template(p).module.test_macro("John")
    assert 'Hello lil John' == out


def test_make_dirs(tmpdir):
    p = tmpdir.mkdir("sub")

    assert str(p) == utils.make_dirs(p)


def test_df_read_multi(tmpdir):
    # I should refactor this test with a list or something using functools import partial for the csv case
    p = tmpdir.mkdir("sub")
    df_test = pd.DataFrame(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c']
    )
    # test for csv
    fn = p.join("test.csv")
    df_test.to_csv(fn, index=False)
    assert df_test.equals(utils.df_read_multi(fn))
    # test for feather
    fn = p.join("test.feather")
    df_test.to_feather(fn)
    pd.read_feather(fn)
    assert df_test.equals(utils.df_read_multi(fn))
    # test for pickle
    fn = p.join("test.pickle")
    df_test.to_pickle(fn)
    pd.read_pickle(fn)
    assert df_test.equals(utils.df_read_multi(fn))

    with pytest.raises(ValueError):
        fn = p.join("test.test")
        fn.write("the sins of the father")
        utils.df_read_multi(fn)


def test_df_to_multi(tmpdir):
    p = tmpdir.mkdir("sub")
    df_test = pd.DataFrame(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c']
    )
    # test for csv
    fn = p.join('test.csv')
    utils.df_to_multi(df_test, fn)
    df = pd.read_csv(fn, index_col=False)
    assert df.equals(df_test)
    # test for feather
    fn = p.join('test.feather')
    utils.df_to_multi(df_test, fn)
    df = pd.read_feather(fn)
    assert df.equals(df_test)
    # test for csv
    fn = p.join('test.pickle')
    utils.df_to_multi(df_test, fn)
    df = pd.read_pickle(fn)
    assert df.equals(df_test)

    with pytest.raises(ValueError):
        fn = p.join("test.test")
        utils.df_to_multi(df_test, fn)


def test_convert_to_snake_case():
    test_str = 'meme_review_best_news_source'
    assert test_str == utils.convert_to_snake_case("MemeReviewBestNewsSource")


def test_range_datetime():
    datetime_start = datetime.datetime(2019, 1, 15)
    datetime_end = datetime.datetime(2019, 1, 20)
    range_lst_no_offset = [
        datetime.datetime(2019, 1, 15, 0, 0),
        datetime.datetime(2019, 1, 16, 0, 0),
        datetime.datetime(2019, 1, 17, 0, 0),
        datetime.datetime(2019, 1, 18, 0, 0),
        datetime.datetime(2019, 1, 19, 0, 0),
        datetime.datetime(2019, 1, 20, 0, 0),
    ]
    range_lst_w_offset = [
        datetime.datetime(2019, 1, 15, 0, 0),
        datetime.datetime(2019, 1, 17, 0, 0),
        datetime.datetime(2019, 1, 19, 0, 0),
    ]

    drange = utils.range_datetime(datetime_start, datetime_end)
    assert range_lst_no_offset == list(drange)
    drange = utils.range_datetime(datetime_start, datetime_end, offsets.Day(2))
    assert range_lst_w_offset == list(drange)


def test_get_first_fortnight_last_day():
    day = utils.get_first_fortnight_last_day(datetime.datetime(2019, 1, 17))
    assert datetime.datetime(2019, 1, 14) == day


def test_normalize_arr():
    arr = utils.normalize_arr(np.array([2, 5, 7, 6, 5]))
    arr_test = np.array([0.08, 0.2, 0.28, 0.24, 0.2])
    assert np.array_equal(arr, arr_test)


def test_apply_time_bounds():
    sd = '2019-01-02'
    ed = '2019-01-03'
    df_test = pd.DataFrame(
        np.array([[4, 5, 6], [7, 8, 9]]),
        index=pd.date_range(start='1/02/2019', end='1/03/2019', freq='D'),
    )
    df = pd.DataFrame(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [3, 2, 1], [6, 5, 4]]),
        index=pd.date_range(start='1/1/2019', end='1/05/2019', freq='D'),
    )
    assert utils.apply_time_bounds(df, sd, ed, None).equals(df_test)

    df['date'] = pd.date_range(start='1/1/2019', end='1/05/2019', freq='D')
    df_test['date'] = pd.date_range(start='1/02/2019', end='1/03/2019', freq='D')

    assert utils.apply_time_bounds(df, sd, ed, 'date').equals(df_test)


def test_normalize_ds_index():
    # test ds_col in df. ds_col in index and ds_col not in ds
    df = pd.DataFrame(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c']
    )

    assert df.equals(utils.normalize_ds_index(df, 'b'))

    df.index = df.index.rename('PLUS_ULTRA')

    with pytest.raises(ValueError):
        utils.normalize_ds_index(df, 'not')


def test_standarize_values():
    sr = pd.Series([2, 4, 6, 8, 10])
    sr_test = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0])
    test_ceros = pd.Series([0, 0, 0, 0, 0, 0])

    assert sr_test.equals(utils.standarize_values(sr))
    assert test_ceros.equals(utils.standarize_values(test_ceros))

    with pytest.raises(TypeError):
        utils.standarize_values([1, 2, 3, 4, 5])


def test_robust_standarize_values():
    df_test = pd.Series(np.linspace(-1.0, 1.0, num=5))

    assert df_test.equals(utils.robust_standarize_values(pd.Series([2, 4, 6, 8, 10])))
    with pytest.raises(TypeError):
        utils.robust_standarize_values([1, 2, 3, 4, 5])


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (["walk", "away", "my", "boy"], "('walk','away','my','boy')"),
        (["walk", "away", "my", "boy", 4], "('walk','away','my','boy',4)"),
        ([1, 3, 4.23], "(1,3,4.23)"),
        ([], "()"),
    ],
)
def test_format_in_clause(test_input, expected):
    """Test valid calls to format_in_clause (mixed and empty)"""
    assert expected == utils.format_in_clause(test_input)


@pytest.mark.parametrize(
    "test_input",  # noqa
    [[sum, "hello"], [1, object()], [1, ["walk", "away", "my", "boy"]]],
)
def test_format_in_clause_invalid_type(test_input):
    """Test iterable formatting with invalid value types."""
    with pytest.raises(utils.BadInClauseException):
        utils.format_in_clause(test_input)


def test_get_cloudera_sql_stats_aggr():
    str_test1 = """SUM(plz kill me) AS sum_None,
AVG(plz kill me) AS mean_None,
APPX_MEDIAN(plz kill me) AS median_None,"""
    str_test2 = """SUM(plz kill me) AS sum_tiny_Rick,
AVG(plz kill me) AS mean_tiny_Rick,
APPX_MEDIAN(plz kill me) AS median_tiny_Rick,"""
    str_test3 = """SUM(plz kill me) AS sum_None,
AVG(plz kill me) AS mean_None,
APPX_MEDIAN(plz kill me) AS median_None,
MIN(plz kill me) AS min_None,
MAX(plz kill me) AS max_None,"""
    str_test4 = """SUM(plz kill me) AS sum_None,
AVG(plz kill me) AS mean_None,
APPX_MEDIAN(plz kill me) AS median_None,
STDDEV(plz kill me) AS std_None,"""
    str_test5 = """SUM(plz kill me) AS sum_None,
AVG(plz kill me) AS mean_None,
APPX_MEDIAN(plz kill me) AS median_None,
NDV(plz kill me) AS unique_None,"""
    str_test6 = """SUM(plz kill me) AS sum_None,
AVG(plz kill me) AS mean_None,
APPX_MEDIAN(plz kill me) AS median_None,
COUNT(plz kill me) AS count_rows_None,"""
    str_test7 = """SUM(plz kill me) AS sum_None,
AVG(plz kill me) AS mean_None,
APPX_MEDIAN(plz kill me) AS median_None"""

    assert str_test1 == utils.get_cloudera_sql_stats_aggr("plz kill me")
    assert str_test2 == utils.get_cloudera_sql_stats_aggr("plz kill me", "tiny_Rick")
    assert str_test3 == utils.get_cloudera_sql_stats_aggr(
        "plz kill me", with_minmax=True
    )
    assert str_test4 == utils.get_cloudera_sql_stats_aggr("plz kill me", with_std=True)
    assert str_test5 == utils.get_cloudera_sql_stats_aggr("plz kill me", with_ndv=True)
    assert str_test6 == utils.get_cloudera_sql_stats_aggr(
        "plz kill me", with_count=True
    )
    assert str_test7 == utils.get_cloudera_sql_stats_aggr(
        "plz kill me", ends_comma=False
    )


def test_get_cloudera_sample_cut():
    int_test_none = 9_223_372_036_854_775_807
    int_test_not_none = 18_446_744_073_709_551_615

    assert int_test_none == utils.get_cloudera_sample_cut(None)
    assert int_test_not_none == utils.get_cloudera_sample_cut(2)


def test_get_cloudera_hashed_sample_clause():
    str_test = 'AND abs(fnv_hash(CAST(34 AS bigint))) <= 4611686018427387903'
    # utils.get_cloudera_hashed_sample_clause(34,69) # This logic should be changed

    assert str_test == utils.get_cloudera_hashed_sample_clause(34, 0.5)


def test_str_normalize_pandas():
    # testing if lst returns as lst_test and the replace with kwargs.
    lst = ["HeLLo", "darkñéSS", "mÿ", "odd", "frieñd"]
    lst_test = ["hello", "darkness", "my", "odd", "friend"]
    lst_test_repl = ["hello", "darkness", "my", "old", "friend"]
    kwargs = {'pat': 'odd', 'repl': 'old'}

    assert pd.DataFrame(lst_test).equals(utils.str_normalize_pandas(pd.DataFrame(lst)))
    assert pd.DataFrame(lst_test_repl).equals(
        utils.str_normalize_pandas(pd.DataFrame(lst), kwargs)
    )
    assert pd.Series(lst_test).equals(utils.str_normalize_pandas(pd.Series(lst)))
    assert pd.Series(lst_test_repl).equals(
        utils.str_normalize_pandas(pd.Series(lst), kwargs)
    )
    with pytest.raises(TypeError):
        utils.str_normalize_pandas(22)


def test_df_optimize_float_types():
    df64 = pd.DataFrame([2.0, 4.0, 5.0, 6.0, 8.0, 10.0], dtype='float64')
    df32 = pd.DataFrame([2.0, 4.0, 5.0, 6.0, 8.0, 10.0], dtype='float32')
    type16_test = pd.DataFrame([2.0, 4.0, 5.0, 6.0, 8.0, 10.0], dtype='float16')

    assert type16_test.dtypes.equals(utils.df_optimize_float_types(df64).dtypes)
    assert type16_test.dtypes.equals(utils.df_optimize_float_types(df32).dtypes)


def test_df_replace_empty_strs_null():
    df = pd.DataFrame([' ', " ", "batman!"])
    df_test = pd.DataFrame([np.nan, np.nan, "batman!"])

    assert df_test.equals(utils.df_replace_empty_strs_null(df))


def test_df_drop_nulls():
    df = pd.DataFrame(
        {'a': [np.nan, np.nan, "batman!"], 'b': [1, 2, 3], 'c': ["", "", "lider!"]}
    )

    df_test = pd.DataFrame({'b': [1, 2, 3]})
    assert df_test.equals(utils.df_drop_nulls(df.copy()))

    df_test = pd.DataFrame({'b': [1, 2, 3], 'c': [np.nan, np.nan, "lider!"]})
    assert df_test.equals(utils.df_drop_nulls(df.copy(), protected_cols=['c']))


def test_df_drop_std():
    df = pd.DataFrame({'a': [0.01, 0.012, 0.013], 'b': [1, 1.2, 1.3], 'c': [2, 2, 3]})
    df_test = pd.DataFrame({'b': [1, 1.2, 1.3], 'c': [2, 2, 3]})
    df_test_diff_std = pd.DataFrame({'c': [2, 2, 3]})
    df_test_protected = pd.DataFrame(
        {'a': [0.01, 0.012, 0.013], 'b': [1, 1.2, 1.3], 'c': [2, 2, 3]}
    )

    assert df_test.equals(utils.df_drop_std(df.copy()))
    assert df_test_diff_std.equals(utils.df_drop_std(df.copy(), min_std_dev=0.5))
    assert df_test_protected.equals(utils.df_drop_std(df, protected_cols=['a']))


# [NTH] it will be cool to use some normal distribution for the df
def test_df_drop_corr():
    df = pd.DataFrame(
        [(0.4, 0.3, 0.2), (0.4, 0.6, 0.3), (0.4, 0.0, 0.2), (0.7, 0.1, 0.5)],
        columns=['a', 'b', 'c'],
    )
    df_test = pd.DataFrame(index=[0, 1, 2, 3])

    with pytest.raises(ValueError):
        utils.df_drop_corr(df.copy(), '34')

    df_test.equals(utils.df_drop_corr(df.copy(), 'a', frac=1, random_state=42))

    df_test = pd.DataFrame(data={'c': [0.2, 0.3, 0.2, 0.5]})
    df_test.equals(
        utils.df_drop_corr(df, 'a', protected_cols=['c'], frac=1, random_state=42)
    )


def test_df_get_typed_cols():
    df = pd.DataFrame(
        {
            'a': [0.01, 0.012, 0.013],
            'b': [True, True, False],
            'c': [
                datetime.datetime(2019, 10, 25),
                datetime.datetime(2019, 10, 26),
                datetime.datetime(2019, 10, 27),
            ],
            'd': ['bip', "bip", "Ritchie"],
        }
    )

    assert pd.Series(utils.df_get_typed_cols(df, col_type='num')).equals(
        pd.Series(['a'])
    )
    assert pd.Series(utils.df_get_typed_cols(df, col_type='bool')).equals(
        pd.Series(['b'])
    )
    assert pd.Series(utils.df_get_typed_cols(df, col_type='date')).equals(
        pd.Series(['c'])
    )
    assert pd.Series(utils.df_get_typed_cols(df)).equals(pd.Series(['d']))


def test_df_encode_categorical_dummies():
    # df, cat_cols=[], skip_cols=[], top=25, other_val='OTHER'
    df = pd.DataFrame(
        [(0.4, "s", 0.2), (0.4, "m", 0.3), (0.4, "h", 0.2)], columns=['a', 'b', 'c']
    )

    df_test = pd.DataFrame(
        [[0.4, 0.2, 0, 0, 1], [0.4, 0.3, 0, 1, 0], [0.4, 0.2, 1, 0, 0]],
        columns=['a', 'c', 'b_h', 'b_m', 'b_s'],
    )
    df_test = df_test.astype(
        dtype={
            'a': 'float64',
            'c': 'float64',
            'b_h': 'uint8',
            'b_m': 'uint8',
            'b_s': 'uint8',
        }
    )
    dummy_test = np.array(['b_m', 'b_h', 'b_s'])

    df_res, dummy = utils.df_encode_categorical_dummies(df.copy(), ['b'])
    assert df_test.equals(df_res)
    assert np.array_equal(np.sort(dummy_test), np.sort(np.array(dummy)))

    df_res, dummy = utils.df_encode_categorical_dummies(df.copy(), ['b'], ['b'])

    assert df_res.equals(df)
    assert not dummy


def test_df_drop_single_factor_level():
    df = pd.DataFrame([(0.4, "s", 0.2), (0.4, "", 0.3)], columns=['a', 'b', 'c'])
    df_test = pd.DataFrame([(0.4, 0.2), (0.4, 0.3)], columns=['a', 'c'])

    assert df_test.equals(utils.df_drop_single_factor_level(df))


@pytest.mark.parametrize(
    "example_input, expected",
    [
        # TODO March 19, 2019: add parametrized raises, might be like:
        # ('test', pytest.raises(AssertionError)),
        ([], []),
        ([4, 3, 2, 2, 1], [4, 3, 2, 1]),
        ([1, 1, 1], [1]),
        (['a', 'aa', []], ['a', 'aa', []]),
    ],
)
def test_dedup_list(example_input, expected):
    assert utils.dedup_list(example_input) == expected
    with pytest.raises(AssertionError):
        utils.dedup_list("a")


@pytest.mark.parametrize(
    "regex, cols_list_test, expected",
    [
        ([r'ah'], ['a', 'al', 'alp', 'alph', 'alpha', 'alp_ha'], []),
        ([r'rav'], ['b', 'br', 'bra', 'brav', 'bravo', 'bra_vo'], ['brav', 'bravo']),
    ],
)
def test_get_matching_columns(cols_list_test, regex, expected):
    assert utils.get_matching_columns(cols_list_test, regex) == expected


@pytest.mark.parametrize(
    "cols, include_regexes, exclude_regexes, expected",
    [
        (
            ['a', 'al', 'alp', 'alph', 'alpha', 'alp_ha'],
            ['lph'],
            ['_'],
            ['alph', 'alpha'],
        ),
        (
            ['b', 'br', 'bra', 'brav', 'bravo', 'bra_vo'],
            ['bra'],
            ['o'],
            ['bra', 'brav'],
        ),
        (
            ['b', 'br', 'bra', 'brav', 'bravo', 'bra_vo'],
            ['b'],
            ['b'],
            [],
        ),  # regexes overlap
    ],
)
def test_get_include_exclude_columns(cols, include_regexes, exclude_regexes, expected):
    assert (
        utils.get_include_exclude_columns(cols, include_regexes, exclude_regexes)
        == expected
    )


@pytest.mark.parametrize(
    "cols, include_regexes, exclude_regexes, expected_error",
    [([], ['alpha'], ['bravo'], ValueError)],
)
def test_get_include_exclude_columns_empty_cols_list(
    cols, include_regexes, exclude_regexes, expected_error
):
    with pytest.raises(expected_error):
        utils.get_include_exclude_columns(cols, include_regexes, exclude_regexes)


# [WONT DO]
# def test_local_df_cache():
#     pass


@pytest.mark.parametrize(
    "ed, tw, fw, expected",
    [
        (
            '2019-10-25',
            2,
            3,
            (
                pd.Timestamp('2019-10-23'),
                datetime.datetime(2019, 10, 25),
                pd.Timestamp('2019-10-28'),
            ),
        ),
        (
            '2018-01-01',
            60,
            30,
            (
                pd.Timestamp('2017-11-02'),
                datetime.datetime(2018, 1, 1),
                pd.Timestamp('2018-01-31'),
            ),
        ),
        # will fail due to raise on negative windows
        pytest.param('2019-10-25', -2, 1, (1, 1, 1), marks=pytest.mark.xfail),
        pytest.param('2019-10-25', 2, -1, (1, 1, 1), marks=pytest.mark.xfail),
    ],
)
def test_create_forecaster_dates(ed, tw, fw, expected):
    # Testing various input dates and windows
    assert utils.create_forecaster_dates(ed, tw, fw) == expected


def test_same_dataframe():
    df_x = get_first_df_diff_deviations_functions()
    df_y = get_first_df_diff_deviations_functions()
    rv = utils.dataframe_diff(df_x, df_y, ['date_col'])
    assert len(rv[0]) == 0 & len(rv[1]) == 0


def test_diff_different_df_without_add():
    df_x = get_first_df_diff_deviations_functions()
    df_y = get_second_df_diff_deviations_functions()
    rv = utils.dataframe_diff(df_x, df_y, ['date_col'])
    assert len(rv[1]) == 0


def test_first_output_diff_different_df_without_add():
    df_x = get_first_df_diff_deviations_functions()
    df_y = get_second_df_diff_deviations_functions()
    rv = utils.dataframe_diff(df_x, df_y, ['date_col'])
    first_rv_mock = pd.DataFrame(
        {
            'date_col': {
                0: '2020-11-11',
                1: '2020-11-12',
                2: '2020-11-13',
                3: '2020-11-14',
            },
            'value_x': {0: 20, 1: 100, 2: 10, 3: 1},
            'value_y': {0: 10, 1: 20, 2: 20, 3: 10},
            'column_name': {
                0: 'row_count',
                1: 'row_count',
                2: 'row_count',
                3: 'row_count',
            },
        }
    )
    assert_frame_equal(rv[0], first_rv_mock)


def test_additional_case_first_output():
    df_x = get_first_df_diff_deviations_functions()
    df_y = pd.DataFrame(
        {
            'date_col': [
                '2020-11-11',
                '2020-11-12',
                '2020-11-13',
                '2020-11-14',
                '2020-11-15',
            ],
            'row_count': [10, 20, 20, 10, 10],
        }
    )
    first_rv_mock = pd.DataFrame(
        {
            'date_col': {
                0: '2020-11-11',
                1: '2020-11-12',
                2: '2020-11-13',
                3: '2020-11-14',
            },
            'value_x': {0: 20, 1: 100, 2: 10, 3: 1},
            'value_y': {0: 10, 1: 20, 2: 20, 3: 10},
            'column_name': {
                0: 'row_count',
                1: 'row_count',
                2: 'row_count',
                3: 'row_count',
            },
        }
    )
    rv = utils.dataframe_diff(df_x, df_y, ['date_col'])
    assert_frame_equal(rv[0], first_rv_mock)


def test_additional_case_second_output():
    df_x = get_first_df_diff_deviations_functions()
    df_y = pd.DataFrame(
        {
            'date_col': [
                '2020-11-11',
                '2020-11-12',
                '2020-11-13',
                '2020-11-14',
                '2020-11-15',
            ],
            'row_count': [10, 20, 20, 10, 10],
        }
    )

    second_rv_mock = pd.DataFrame(
        {'date_col': {0: '2020-11-15'}, 'row_count': {0: 10}, 'sets': {0: 'df_y'}}
    )
    rv = utils.dataframe_diff(df_x, df_y, ['date_col'])
    assert_frame_equal(rv[1], second_rv_mock)


def test_first_output_diff_diferent_df_reverse_case():
    df_x = get_first_df_diff_deviations_functions()
    df_y = get_second_df_diff_deviations_functions()
    rv = utils.dataframe_diff(df_x, df_y, ['date_col'])
    rv_reverse = utils.dataframe_diff(df_y, df_x, ['date_col'])
    assert (list(rv[0].value_x) == list(rv_reverse[0].value_y)) & (
        list(rv[0].value_y) == list(rv_reverse[0].value_x)
    )


def test_second_output_diff_diferent_df_reverse_case():
    df_x = get_first_df_diff_deviations_functions()
    df_y = pd.DataFrame(
        {
            'date_col': [
                '2020-11-11',
                '2020-11-12',
                '2020-11-13',
                '2020-11-14',
                '2020-11-15',
            ],
            'row_count': [10, 20, 20, 10, 10],
        }
    )
    rv = utils.dataframe_diff(df_x, df_y, ['date_col'])
    rv_reverse = utils.dataframe_diff(df_y, df_x, ['date_col'])
    assert_frame_equal(
        rv[1][['date_col', 'row_count']], rv_reverse[1][['date_col', 'row_count']]
    )


def test_len_dev_log_normal_case():
    assert len(
        utils.compute_differences_dataframes(
            get_first_df_diff_deviations_functions(),
            get_second_df_diff_deviations_functions(),
            ["date_col"],
            "_first",
            "_second",
        )
        == 4
    )


def test_dev_log_normal_case():
    assert_frame_equal(
        utils.compute_differences_dataframes(
            get_first_df_diff_deviations_functions(),
            get_second_df_diff_deviations_functions(),
            ["date_col"],
            "_first",
            "_second",
        ),
        pd.DataFrame(
            {
                'date_col': {
                    0: '2020-11-11',
                    1: '2020-11-12',
                    2: '2020-11-13',
                    3: '2020-11-14',
                },
                'row_count_first': {0: 20, 1: 100, 2: 10, 3: 1},
                'row_count_second': {0: 10, 1: 20, 2: 20, 3: 10},
                'diff': {0: 10, 1: 80, 2: -10, 3: -9},
                'diff_%': {0: 50.0, 1: 80.0, 2: -100.0, 3: -900.0},
            }
        ),
    )


def test_numpy_temp_seed():
    # Sets a fixed seed
    fixed_seed = 42
    # Generates a random value without a fixed seed
    random_value_1_without_fixed_seed = np.random.rand()
    # Generates a random value with a fixed seed
    with utils.numpy_temp_seed(seed=fixed_seed):
        random_value_1_with_fixed_seed = np.random.rand()
    # Generates another random value without a fixed seed
    random_value_2_without_fixed_seed = np.random.rand()
    # Generates another random value with a fixed seed
    with utils.numpy_temp_seed(seed=fixed_seed):
        random_value_2_with_fixed_seed = np.random.rand()
    # Generates another random value without a fixed seed
    random_value_3_without_fixed_seed = np.random.rand()

    # Asserts if random values genereated with a fixed seed are equal
    assert random_value_1_with_fixed_seed == random_value_2_with_fixed_seed
    # Asserts if random values genereated without a fixed seed are different
    assert (
        (random_value_1_without_fixed_seed != random_value_2_without_fixed_seed)
        & (random_value_2_without_fixed_seed != random_value_3_without_fixed_seed)
        & (random_value_1_without_fixed_seed != random_value_3_without_fixed_seed)
    )
    # Asserts if random values generated without a fixed seed
    # are different than those generated with a fixed one
    assert (random_value_1_without_fixed_seed != random_value_1_with_fixed_seed) & (
        random_value_1_without_fixed_seed != random_value_2_with_fixed_seed
    )
    assert (random_value_2_without_fixed_seed != random_value_1_with_fixed_seed) & (
        random_value_2_without_fixed_seed != random_value_2_with_fixed_seed
    )
    assert (random_value_3_without_fixed_seed != random_value_1_with_fixed_seed) & (
        random_value_3_without_fixed_seed != random_value_2_with_fixed_seed
    )


def test_ab_split(sample_df):
    expected_dist = round(np.random.uniform(0, 1), 2)
    expected_dist_err = 0.05
    sample_df['test_group'] = sample_df.id.apply(
        lambda id: utils.ab_split(id, 'E1F53135E559C253', expected_dist)
    )
    split_dist = 1 - np.mean(sample_df.test_group.astype(int))
    assert (
        expected_dist + expected_dist_err >= split_dist
        and expected_dist - expected_dist_err <= split_dist
    )


def test_col_sample_display(sample_df):
    with mock.patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
        utils.col_sample_display(sample_df, 'batman', top_val=0.35)
        assert 'Col is batman' in mock_stdout.getvalue()
        assert 'Null count is 0, Null percentage is: 0.00%' in mock_stdout.getvalue()
        assert '4 [3 4 1 2]' in mock_stdout.getvalue()
    with mock.patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
        utils.col_sample_display(sample_df, 'batman', quantile=0.25)
        assert 'Col is batman' in mock_stdout.getvalue()
        assert 'Null count is 0, Null percentage is: 0.00%' in mock_stdout.getvalue()
        assert '4 [3 4 1 2]' in mock_stdout.getvalue()
    with mock.patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
        utils.col_sample_display(
            df=sample_df, col='two_face', top_val=0.35, num_sample=15000
        )
        assert 'Col is two_face' in mock_stdout.getvalue()
        assert 'Null count is 0, Null percentage is: 0.00%' in mock_stdout.getvalue()


def test_sum_count_aggregation(sample_df):
    expected_df = pd.DataFrame(
        {
            'robin_count': {1: 2536, 2: 2486, 3: 2477, 4: 2501},
            'robin_count_perc': {1: 0.2536, 2: 0.2486, 3: 0.2477, 4: 0.2501},
        }
    )
    agg_df = utils.sum_count_aggregation(sample_df, ['batman'], ['robin'], ['count'])
    assert expected_df.equals(agg_df)


def test_sum_count_time_series(sample_df, sample_timeseries_df):
    df = utils.sum_count_time_series(sample_df, 'two_face', ['batman', 'robin'])
    assert df.equals(sample_timeseries_df)


def test_category_reductor(sample_df):
    df = utils.category_reductor(sample_df, 'robin').copy()
    assert df.describe()['unique'] == 8


def test_load_sql_query(tmp_path):
    sql_tpl = (
        "SELECT age, favorite_food\n"
        "FROM super_heroes\n"
        "WHERE hero = {{ hero }}\n"
        "AND hero_version = {{ version }}\n"
    )
    sql_path = tmp_path / 'TestUtils.load_sql_query.sql'
    sql_path.write_text(sql_tpl)
    sql_fn = str(sql_path)
    assert utils.load_sql_query(sql_fn, {'hero': 'batman', 'version': 'iron'}) == (
        "SELECT age, favorite_food\n"
        "FROM super_heroes\n"
        "WHERE hero = batman\n"
        "AND hero_version = iron"
    )


def test_get_sql_stats_aggr():
    assert utils.get_sql_stats_aggr(
        'batman', as_name='batman', with_std=True, with_ndv=True, with_count=True
    ) == (
        "\n    "
        "SUM(batman) as sum_batman,\n    "
        "AVG(batman) as mean_batman,\n    "
        "APPX_MEDIAN(batman) as median_batman,\n    "
        "MIN(batman) as min_batman,\n    "
        "MAX(batman) as max_batman,\n "
        "STDDEV(batman) as std_batman,\n "
        "NDV(batman) as unique_batman,\n "
        "COUNT(1) as count_batman,"
    )


def test_get_null_count_aggr():
    value_list = ['robin', 'batman']
    assert utils.get_null_count_aggr(
        value_list, no_ending_comma=True, empty_string_null=True
    ) == (
        "SUM( CASE WHEN robin = ''\n    "
        "THEN 1 ELSE 0 END ) AS null_countrobin,\n"
        "SUM( CASE WHEN batman = ''\n    "
        "THEN 1 ELSE 0 END ) AS null_countbatman"
    )


def test_get_sqlserver_hashed_sample_clause():
    assert utils.get_sqlserver_hashed_sample_clause(123456, 0.5) == (
        "\n    "
        "AND ABS(CAST(HASHBYTES('SHA1',\n        "
        "123456) AS BIGINT)) % 100 <= 50"
    )
