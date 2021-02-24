from unittest.mock import patch, MagicMock, ANY

import pandas as pd
from pandas._testing import assert_frame_equal

from muttlib.dbconn import OracleClient
import pytest


@pytest.fixture()
def oracle_client():
    client = OracleClient(
        username="scott",
        host="127.0.0.1",
        port=1521,
        dialect="oracle",
        password="tiger",
    )

    return client


@pytest.fixture(scope='session')
def oracle_client_with_schema():
    client = OracleClient(
        username="scott",
        host="127.0.0.1",
        port=1521,
        dialect="oracle",
        password="tiger",
        schema='test',
    )

    return client


def url_connection(oracle_client):
    engine = oracle_client.get_engine()
    assert str(engine.url) == 'oracle://scott:tiger@127.0.0.1:1521'


def url_connection_with_scheme(oracle_client_with_schema):
    engine = oracle_client_with_schema.get_engine()
    assert str(engine.url) == 'oracle://scott:tiger@127.0.0.1:1521/test'


def url_connection_with_scheme_error(oracle_client):
    engine = oracle_client.get_engine()
    assert engine.url != 'oracle://scott:tiger@127.0.0.1:1521/test'


def test_insert_from_frame_connects_fix_clobs_false(oracle_client):

    with patch("muttlib.dbconn.base.create_engine") as create_engine:

        df = MagicMock(dtypes='test')

        table = "test_table"
        oracle_client.insert_from_frame(df, table, fix_clobs=False)
        create_engine.assert_called_once_with(
            oracle_client.conn_str, connect_args=ANY, echo=ANY
        )
        df.to_sql.assert_called_once_with(
            table,
            create_engine.return_value.connect.return_value.__enter__.return_value,
            if_exists='append',
            dtype='test',
            index=False,
        )


def test_insert_from_frame_connects_fix_clobs_true(oracle_client):

    with patch("muttlib.dbconn.base.create_engine") as create_engine, patch(
        "muttlib.dbconn.oracle.VARCHAR"
    ) as varchar:

        df = MagicMock(dtypes='object')
        df.columns.__getitem__.return_value.tolist.return_value = ["col1", "col2"]
        df.__getitem__.return_value.str.len.return_value.max.return_value = 3

        table = "test_table"
        oracle_client.insert_from_frame(df, table, fix_clobs=True)

        create_engine.assert_called_once_with(
            oracle_client.conn_str, connect_args=ANY, echo=ANY
        )

        df.to_sql.assert_called_once_with(
            table,
            create_engine.return_value.connect.return_value.__enter__.return_value,
            if_exists='append',
            dtype={'col1': varchar(3), 'col2': varchar(3)},
            index=False,
        )


def test_insert_from_frame_connects_upper_case_cols_true(oracle_client):

    with patch("muttlib.dbconn.base.create_engine") as create_engine:

        df = pd.DataFrame({'col1': [1], 'col2': [3.0]})

        table = "test_table"
        oracle_client.insert_from_frame(df, table, fix_clobs=True, upper_case_cols=True)

        assert_frame_equal(df, pd.DataFrame({'COL1': [1], 'COL2': [3.0]}))


def test_insert_from_frame_connects_upper_case_cols_false(oracle_client):

    with patch("muttlib.dbconn.base.create_engine") as create_engine:

        df = pd.DataFrame({'col1': [1], 'col2': [3.0]})

        table = "test_table"
        oracle_client.insert_from_frame(
            df, table, fix_clobs=True, upper_case_cols=False
        )

        assert_frame_equal(df, pd.DataFrame({'col1': [1], 'col2': [3.0]}))
