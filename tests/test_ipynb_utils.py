"""muttlib.ipynb_utils test suite."""
import io
import sys
import subprocess

import numpy
import pandas
import pytest

from muttlib import ipynb_utils as utils
from muttlib.utils import numpy_temp_seed
from unittest import mock


@pytest.fixture()
def sample_df():
    base_date = numpy.datetime64('2020-01-01')
    n_rows = 100 ** 2
    with numpy_temp_seed():
        df = pandas.DataFrame(
            {
                'id': numpy.arange(n_rows),
                'batman': numpy.random.randint(1, 5, n_rows),
                'robin': numpy.random.randint(1, 1000, n_rows),
            }
        )
        df['str_robin'] = df.robin.astype(str)
        df['riddler'] = df.batman.replace([1, 2, 3, 4], ['a', 'b', 'c', 'd'])
        df['two_face'] = df.batman.apply(lambda x: base_date + numpy.random.choice(x))
    return df


@pytest.fixture()
def sample_timeseries_df():
    return pandas.DataFrame(
        {
            'two_face': {
                0: pandas.Timestamp('2020-01-01 00:00:00'),
                1: pandas.Timestamp('2020-01-02 00:00:00'),
                2: pandas.Timestamp('2020-01-03 00:00:00'),
                3: pandas.Timestamp('2020-01-04 00:00:00'),
            },
            'batman_sum': {0: 9812, 1: 7703, 2: 5024, 3: 2404},
            'batman_count': {0: 5173, 1: 2764, 2: 1462, 3: 601},
            'robin_sum': {0: 2617661, 1: 1374750, 2: 726399, 3: 297004},
            'robin_count': {0: 5173, 1: 2764, 2: 1462, 3: 601},
        }
    )


def test_convert_to_snake_case():
    cases = [
        'Batman and Robin',
        'BatmanAndRobin',
        'Batman_and_Robin',
        'Batman-and-Robin',
        ' Batman and Robin',
        ' Batman and Robin ',
        'Batman and Robin ',
        'Batman  and   Robin  ',
    ]
    for raw_string in cases:
        assert utils.convert_to_snake_case(raw_string) == 'batman_and_robin'


def test_list_to_sql_tuple():
    param_list = ['batman', 'and', 'robin']
    assert utils.list_to_sql_tuple(param_list) == '(batman, and, robin)'


def test_describe_table():
    db_connector_mock = mock.Mock()
    db_connector_mock.configure_mock(**{'execute.return_value': 'OK'})
    utils.describe_table('super_heroes', db_connector_mock)
    db_connector_mock.execute.assert_called_once_with('describe super_heroes')


def test_write_to_clipboard():
    with mock.patch('subprocess.Popen') as mock_subproc_popen:
        process_mock = mock.Mock()
        attrs = {'communicate.return_value': ('batman', 'robin')}
        process_mock.configure_mock(**attrs)
        mock_subproc_popen.return_value = process_mock
        utils.write_to_clipboard('batman')
        mock_subproc_popen.assert_called_once_with(
            'pbcopy', env={'LANG': 'en_US.UTF-8'}, stdin=subprocess.PIPE
        )
        process_mock.communicate.assert_called_once_with('batman'.encode('utf-8'))


def test_list_vals_contains_str():
    value = 'batman'
    value_list = ['robin', 'batman', 'riddler', 'two_face', 'Batman']
    assert utils.list_vals_contains_str(value_list, value) == ['batman', 'Batman']


def test_ab_split(sample_df):
    expected_dist = round(numpy.random.uniform(0, 1), 2)
    expected_dist_err = 0.05
    sample_df['test_group'] = sample_df.id.apply(
        lambda id: utils.ab_split(id, 'E1F53135E559C253', expected_dist)
    )
    split_dist = 1 - numpy.mean(sample_df.test_group.astype(int))
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


def test_get_one_to_one_relationship(sample_df):
    df = utils.get_one_to_one_relationship(
        sample_df, factor_id='batman', factor_name='riddler'
    )
    assert df.riddler.unique()[0] == 1


def test_sum_count_aggregation(sample_df):
    expected_df = pandas.DataFrame(
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


def test_clean_numeric_col(sample_df):
    df = sample_df.transform(lambda x: x.astype(numpy.str)).copy()
    batman = utils.clean_numeric_col(df, 'batman')
    assert batman.dtype == numpy.int64


def test_optimize_numeric_types(sample_df):
    df = (
        sample_df[['batman', 'robin']]
        .transform(lambda x: x.astype(numpy.float64))
        .copy()
    )
    df = utils.optimize_numeric_types(df)
    assert df.batman.dtype == numpy.float32
    assert df.robin.dtype == numpy.float32


def test_get_string_named_placeholders():
    string = '{batman} and {robin}'
    assert utils.get_string_named_placeholders(string) == ['batman', 'robin']


def test_load_sql_query(tmp_path):
    sql_tpl = (
        "SELECT age, favorite_food\n"
        "FROM super_heroes\n"
        "WHERE hero = {{ hero }}\n"
        "AND hero_version = {{ version }}\n"
    )
    sql_path = tmp_path / 'TestIpynbUtils.load_sql_query.sql'
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


def test_get_include_exclude_columns():
    columns = [
        'batman_count_perc',
        'batman_count',
        'batman_sum',
        'batman_sum_perc',
        'robin_count_perc',
        'robin_count',
        'robin_sum',
        'robin_sum_perc',
    ]
    include_regexes = [r'._sum$']
    exclude_regexes = [r'^robin_.']
    assert utils.get_include_exclude_columns(
        columns, include_regexes=include_regexes
    ) == ['batman_sum', 'robin_sum']
    assert utils.get_include_exclude_columns(
        columns, exclude_regexes=exclude_regexes
    ) == ['batman_count', 'batman_count_perc', 'batman_sum', 'batman_sum_perc']


def test_get_matching_columns():
    columns = [
        'batman_count_perc',
        'batman_count',
        'batman_sum',
        'batman_sum_perc',
        'robin_count_perc',
        'robin_count',
        'robin_sum',
        'robin_sum_perc',
    ]
    regexes = [r'._count$', r'._sum$']
    assert utils.get_matching_columns(columns, regexes) == [
        'batman_count',
        'robin_count',
        'batman_sum',
        'robin_sum',
    ]


def test_get_sqlserver_hashed_sample_clause():
    assert utils.get_sqlserver_hashed_sample_clause(123456, 0.5) == (
        "\n    "
        "AND ABS(CAST(HASHBYTES('SHA1',\n        "
        "123456) AS BIGINT)) % 100 <= 50"
    )


def test_wrap_list_values_quotes():
    value_list = ['batman', 'and', 'robin']
    assert utils.wrap_list_values_quotes(value_list) == ["'batman'", "'and'", "'robin'"]
