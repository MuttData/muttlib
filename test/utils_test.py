from muttlib import utils
import datetime
import pytest
from collections import OrderedDict, namedtuple, deque
import pandas as pd
import numpy as np
import tempfile # Remember to delete this
import pandas as pd

timestamps_inclause = (pd.Timestamp('2018-06-17 00:00:00'), 
                       pd.Timestamp('2018-10-21 00:00:00'), 
                       pd.Timestamp('2018-08-19 00:00:00'))

@pytest.mark.parametrize("test_input,expected", [
    ("utils.str_to_datetime('2019-10-25 18:35:22')", datetime.datetime(2019, 10, 25, 18, 35, 22)),
    ("utils.str_to_datetime('2019-10-25')", datetime.datetime(2019, 10, 25, 0, 0)),
    ("utils.str_to_datetime('2019-10-25 18:35:22.000333')", datetime.datetime(2019, 10, 25, 18, 35, 22, 333)),
    ("utils.str_to_datetime('18:35:22.000333')", datetime.datetime(1900, 1, 1, 18, 35, 22, 333)),
    ("utils.str_to_datetime('18:35:22')", datetime.datetime(1900, 1, 1, 18, 35, 22)),
    ("utils.str_to_datetime('20191025T18:35:22')", datetime.datetime(2019, 10, 25, 18, 35, 22)),
    ("utils.str_to_datetime('2019-10-25T18:35:22')", datetime.datetime(2019, 10, 25, 18, 35, 22)),
    ("utils.str_to_datetime('20191025')", datetime.datetime(2019, 10, 25, 0, 0)),
    ("utils.str_to_datetime('2019-10-25T18')", datetime.datetime(2019, 10, 25, 18, 0)),
    ("utils.str_to_datetime('201910')", datetime.datetime(2019, 10, 1, 0, 0))
])
def test_str_to_datetime(test_input, expected):
    #Testing all posible datetime formats and the exception for wrong format
    assert eval(test_input) == expected 
    with pytest.raises(ValueError):
        utils.str_to_datetime("25/10/2019")

@pytest.mark.parametrize("test_input,expected", [
    ("namedtuple('mr','meeseeks') ('look at me')", utils.dict_to_namedtuple('mr',{'meeseeks': 'look at me'}))
])
def test_dict_to_namedtuple(test_input, expected):
    assert eval(test_input) == expected 

def test_create_dict_id():
    assert '96107c8ce8' == utils.create_dict_id({'meeseeks': 'look at me'})

def test_get_ordered_factor_levels():
    lst = ['cat', 'dog', 'cat', 'dog', 'cat', 'dog', 'horse', 'dog', 'horse',]
    df = pd.DataFrame({'olmcdonald': lst}, columns=['olmcdonald'])
    df2 = utils.get_ordered_factor_levels(df,'olmcdonald')

    assert np.array_equal(np.array(['dog', 'cat', 'horse']),df2[0])
    assert 3 == df2[1]
    # Testing the other params
    df2 = utils.get_ordered_factor_levels(df,'olmcdonald',2)

    assert np.array_equal(np.array(['dog', 'cat']),df2[0])
    assert 2 == df2[1]

    df2 = utils.get_ordered_factor_levels(df,'olmcdonald',2,4)
    
    assert np.array_equal(np.array(['dog']),df2[0])
    assert 1 == df2[1]

def test_query_yes_no(monkeypatch):
    # Testing the possible defaults and rewriting the input for the valid or invalid inputs
    og = utils.__builtins__["input"]
    with pytest.raises(ValueError):
        utils.query_yes_no("hit or miss?", 's')
    utils.__builtins__["input"] = lambda: ''
    assert False == utils.query_yes_no("hit or miss?")
    utils.__builtins__["input"] = lambda: ''
    assert True == utils.query_yes_no("hit or miss?",'yes')
    utils.__builtins__["input"] = lambda: 'yes'
    assert True == utils.query_yes_no("hit or miss?")
    utils.__builtins__["input"] = lambda: 'y'
    assert True == utils.query_yes_no("hit or miss?")
    utils.__builtins__["input"] = lambda: 'ye'
    assert True == utils.query_yes_no("hit or miss?")
    utils.__builtins__["input"] = lambda: 'no'
    assert False == utils.query_yes_no("hit or miss?")
    utils.__builtins__["input"] = lambda: 'n'
    assert False == utils.query_yes_no("hit or miss?")

    #[TODO]Validate the while. Need to rewrite this shit to rise an exception but idk shit
    #utils.sys.stdout["write"] = raise Exception('shit happended')    

    utils.__builtins__["input"] = og

def test_path_or_string(tmpdir):
    #generate a tmp file for this test
    p = tmpdir.mkdir("sub").join("test.txt")
    p.write("True")
    assert 'True' == utils.path_or_string(p)

def test_hash_str():
    assert '0c5024ed' == utils.hash_str("hit or miss")

def test_deque_to_geo_hierarchy_dict():
    #Testing the creation of the orderedDict for the 4 levels
    lst = [{'level': 'National', 'select_clause': '', 'group_clause': ''}, 
           {'level': 'Provincial', 'select_clause': "existence is pain", 'post_join_select': 'province_name,', 'group_clause': '1,'},
           {'level': 'Departamental', 'select_clause': "existence is pain", 'post_join_select': 'departament_name,', 'group_clause': '2,'},
           {'level': 'Local', 'select_clause':"existence is pain", 'post_join_select': 'locality_name,', 'group_clause': '3,'},
          ]
    deque_lst = deque(lst)
    cont_lst = [('National', {'select_clause': '', 'group_clause': ''}), 
                ('Provincial', {'select_clause': "existence is pain", 'post_join_select': 'province_name,', 'group_clause': '1,'}), 
                ('Departamental', {'select_clause': "existence is pain", 'post_join_select': 'departament_name,', 'group_clause': '2,'}), 
                ('Local', {'select_clause': "existence is pain", 'post_join_select': 'locality_name,', 'group_clause': '3,'})]
    
    assert OrderedDict(cont_lst[:1]) == utils.deque_to_geo_hierarchy_dict(deque_lst,'National')
    assert OrderedDict(cont_lst[:2]) == utils.deque_to_geo_hierarchy_dict(deque_lst,'Provincial')
    assert OrderedDict(cont_lst[:3]) == utils.deque_to_geo_hierarchy_dict(deque_lst,'Departamental')
    assert OrderedDict(cont_lst[:]) == utils.deque_to_geo_hierarchy_dict(deque_lst,'Local')

def test_read_yaml(tmpdir):
    #generate a tmp file for this test
    lst_test = ['meme', 'clap', 'review', 'clap']

    p = tmpdir.mkdir("sub").join("yaml_test.yaml")
    p.write("""
               - meme
               - clap
               - review
               - clap
            """)
    assert lst_test == utils.read_yaml(p)

def test_get_fathers_mothers_kids_day():
    dates = (pd.Timestamp('2019-06-16'),
             pd.Timestamp('2019-10-20'),
             pd.Timestamp('2019-08-18'),
            )
    assert dates == utils.get_fathers_mothers_kids_day(2019)

@pytest.mark.parametrize("test_input,expected", [
    ("utils.is_special_day(datetime.datetime(2018,1,18),timestamps_inclause)",0),
    ("utils.is_special_day(datetime.datetime(2018,6,17),timestamps_inclause)",1),
    ("utils.is_special_day('2018-01-18',timestamps_inclause)",0),
    ("utils.is_special_day('2018-06-17',timestamps_inclause)",1),
])
def test_is_special_day(test_input, expected):
    assert eval(test_input) == expected 

def test_get_friends_day():
    # just pass a year(int) and gives you the "amigos's day
    assert datetime.datetime(2018, 7, 20) == utils.get_friends_day(2018)

def test_get_semi_month_pay_days():
    dates = [pd.Timestamp('2018-02-16 00:00:00'), 
             pd.Timestamp('2018-03-02 00:00:00'), 
             pd.Timestamp('2018-03-16 00:00:00'), 
             pd.Timestamp('2018-03-30 00:00:00')]

    assert dates == utils.get_semi_month_pay_days('2018-01-01','2018-02-28')

def test_df_info_to_str():
    df = pd.DataFrame({'B': [25, 94, 57, 62, 70]}, columns = ['B'])
    str_cmp = """<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5 entries, 0 to 4
Data columns (total 1 columns):
B    5 non-null int64
dtypes: int64(1)
memory usage: 120.0 bytes
"""
    assert str_cmp == utils.df_info_to_str(df)

def test_template(tmpdir):
    tmp_template = "Hello,my name is {{name}} you kill my {{daddy}} prepare to {{acction}}"
    str_template = "Hello,my name is Inigo Montoya you kill my father prepare to die"

    p = tmpdir.mkdir("sub").join("template_test.html")
    p.write(tmp_template)

    assert str_template == utils.template(p).render(name="Inigo Montoya", daddy="father", acction ="die")

# WIP
def test_local_df_cache():
    pass

#[TODO] Need to make special shit for this ones
def test_make_dirs():
    pass

def test_non_empty_dirs():
    pass

def test_render_jinja_template():
    #utils.render_jinja_template()
    pass