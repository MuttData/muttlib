# pylint:disable=W0611, E1101

from muttlib import utils
import datetime
import pytest
from pathlib import Path
from collections import OrderedDict, namedtuple, deque  # noqa: F401
import pandas as pd
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


# WIP


def test_render_jinja_template(tmpdir):
    tmp_template = (
        "Hello,my name is {{name}} you kill my {{daddy}} prepare to {{acction}}"
    )
    params = {'name': 'Inigo Montoya', 'daddy': 'father', 'acction': 'die'}
    str_template = "Hello,my name is Inigo Montoya you kill my father prepare to die"

    p = tmpdir.mkdir("sub").join("template_test.html")
    p.write(tmp_template)

    assert str_template == utils.render_jinja_template(tmp_template, params)


# [TODO] Need to make special shit for this ones
def test_make_dirs(tmpdir):
    p = tmpdir.mkdir("sub")

    assert str(p) == utils.make_dirs(p)


# [WONT DO]
# def test_local_df_cache():
#     pass
