# pylint:disable=W0611, E1101

from muttlib import utils
import datetime
import pytest
from pathlib import Path
from collections import OrderedDict, namedtuple, deque  # noqa: F401
import pandas as pd
from pandas.tseries import offsets
import numpy as np


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


@pytest.mark.parametrize(
    "test_input,expected",
    [('look at me', utils.dict_to_namedtuple('mr', {'meeseeks': 'look at me'}))],
)
def test_dict_to_namedtuple(test_input, expected):
    assert namedtuple('mr', 'meeseeks')(test_input) == expected


def test_create_dict_id():
    assert '96107c8ce8' == utils.create_dict_id({'meeseeks': 'look at me'})


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


def test_read_yaml(tmpdir):
    # generate a tmp file for this test
    lst_test = ['meme', 'clap', 'review', 'clap']

    p = tmpdir.mkdir("sub").join("yaml_test.yaml")
    p.write(
        """
               - meme
               - clap
               - review
               - clap
            """
    )
    assert lst_test == utils.read_yaml(p)


def test_get_fathers_mothers_kids_day():
    dates = (
        pd.Timestamp('2019-06-16'),
        pd.Timestamp('2019-10-20'),
        pd.Timestamp('2019-08-18'),
    )
    assert dates == utils.get_fathers_mothers_kids_day(2019)


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (datetime.datetime(2018, 1, 18), 0),
        (datetime.datetime(2018, 6, 17), 1),
        ('2018-01-18', 0),
        ('2018-06-17', 1),
    ],
)
def test_is_special_day(test_input, expected):
    timestamps_inclause = (
        pd.Timestamp('2018-06-17 00:00:00'),
        pd.Timestamp('2018-10-21 00:00:00'),
        pd.Timestamp('2018-08-19 00:00:00'),
    )
    assert utils.is_special_day(test_input, timestamps_inclause) == expected


def test_get_friends_day():
    # just pass a year(int) and gives you the "amigos's day
    assert datetime.datetime(2018, 7, 20) == utils.get_friends_day(2018)


def test_get_semi_month_pay_days():
    dates = [
        pd.Timestamp('2018-02-16 00:00:00'),
        pd.Timestamp('2018-03-02 00:00:00'),
        pd.Timestamp('2018-03-16 00:00:00'),
        pd.Timestamp('2018-03-30 00:00:00'),
    ]

    assert dates == utils.get_semi_month_pay_days('2018-01-01', '2018-02-28')


def test_df_info_to_str():
    df = pd.DataFrame({'B': [25, 94, 57, 62, 70]}, columns=['B'])
    str_cmp = """<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5 entries, 0 to 4
Data columns (total 1 columns):
B    5 non-null int64
dtypes: int64(1)
memory usage: 120.0 bytes
"""
    assert str_cmp == utils.df_info_to_str(df)


def test_template_basic(tmpdir):
    tmp_template = (
        "Hello,my name is {{name}} you kill my {{daddy}} prepare to {{acction}}"
    )
    str_template = "Hello,my name is Inigo Montoya you kill my father prepare to die"

    p = tmpdir.mkdir("sub").join("template_test.html")
    p.write(tmp_template)

    assert str_template == utils.template(p).render(
        name="Inigo Montoya", daddy="father", acction="die"
    )


def test_template_macros(tmpdir):
    p = tmpdir.mkdir("sub").join("macros_test.html")
    # testing macros
    macro = "{% macro test_macro(name) %}" "Hello lil {{ name }}" "{% endmacro %}"
    p.write(macro)

    out = utils.template(p).module.test_macro("John")
    assert 'Hello lil John' == out


def test_non_empty_dirs(tmpdir):
    p = tmpdir.mkdir("sub").join("test.test")
    p.write("the sins of the father")
    assert [str(Path(p).parent)] == utils.non_empty_dirs(Path(p).parent)


def test_render_jinja_template(tmpdir):
    tmp_template = (
        "Hello,my name is {{name}} you kill my {{daddy}} prepare to {{acction}}"
    )
    params = {'name': 'Inigo Montoya', 'daddy': 'father', 'acction': 'die'}
    str_template = "Hello,my name is Inigo Montoya you kill my father prepare to die"

    p = tmpdir.mkdir("sub").join("template_test.html")
    p.write(tmp_template)

    assert str_template == utils.render_jinja_template(tmp_template, params)


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


def test_wrap_list_values_quotes():
    test_lst = ["'6'", "'7'", "'8'", "'9'", "'1'", "'3'", "'5'"]

    assert test_lst == utils.wrap_list_values_quotes([6, 7, 8, 9, 1, 3, 5])


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

    print(utils.normalize_ds_index(df, 'PLUS_ULTRA'))

    with pytest.raises(ValueError):
        utils.normalize_ds_index(df, 'not')


def test_standarize_ts():
    ts = pd.to_datetime(['1/2/2018', '1/4/2018', '1/6/2018', '1/8/2018', '1/10/2018'])
    lst_test = np.array([-0.0, -0.25, -0.5, -0.75, -1.0])
    test_ceros = pd.Series([0, 0, 0, 0, 0, 0])

    assert test_ceros.equals(utils.standarize_ts(test_ceros))
    assert np.array_equal(np.array(utils.standarize_ts(ts)), lst_test)


def test_robust_standarize_ts():
    df_test = pd.Series(np.linspace(-1.0, 1.0, num=5))

    assert df_test.equals(utils.robust_standarize_ts(pd.Series([2, 4, 6, 8, 10])))


def test_none_or_empty_pandas():
    assert utils.none_or_empty_pandas(None)
    df_test = pd.Series(np.linspace(-1.0, 1.0, num=5))
    assert not utils.none_or_empty_pandas(df_test)
    df_test = pd.DataFrame(np.linspace(-1.0, 1.0, num=5))
    assert not utils.none_or_empty_pandas(df_test)
    with pytest.raises(ValueError):
        assert not utils.none_or_empty_pandas(34)


def test_in_clause_requirement():
    lst_test = [1, 2, 3, 4, 5]
    tppl_test = (1, 2, 3, 4, 5)

    assert utils.in_clause_requirement(lst_test)
    assert utils.in_clause_requirement(tppl_test)
    assert not utils.in_clause_requirement(34)


def test_format_in_clause():
    str_test = "(walk,away,my,boy)"
    with pytest.raises(utils.BadInClauseException):
        utils.format_in_clause("hello")

    assert str_test == utils.format_in_clause(["walk", "away", "my", "boy"])


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
    int_test_none = 9223372036854775807
    int_test_not_none = 18446744073709551615

    assert int_test_none == utils.get_cloudera_sample_cut(None)
    assert int_test_not_none == utils.get_cloudera_sample_cut(2)


def test_get_cloudera_hashed_sample_clause():
    str_test = 'AND abs(fnv_hash(CAST(34 AS bigint))) <= 4611686018427387903'
    # utils.get_cloudera_hashed_sample_clause(34,69) # This logic should be changed

    assert str_test == utils.get_cloudera_hashed_sample_clause(34, 0.5)


# this looks hard should I do it?
# def test_setup_logging():
# pass

# [WONT DO]
# def test_local_df_cache():
#     pass
