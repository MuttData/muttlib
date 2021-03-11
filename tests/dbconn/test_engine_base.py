from unittest.mock import patch, MagicMock, ANY

import pandas as pd
from pandas._testing import assert_frame_equal

from muttlib.dbconn.base import EngineBaseClient, BaseClient
import pytest


@pytest.fixture
def engine_baseClient():
    client = EngineBaseClient(
        database="database",
        host="host",
        password="password",
        port=5544,
        username="username",
        dialect="mysql",
    )

    return client


def test_base_insert_from_frame_connection_not_none(engine_baseClient):
    df = pd.DataFrame({'col1': ['1'], 'col2': ['3.0']})

    with patch("muttlib.dbconn.base.create_engine") as create_engine, patch.object(
        df, 'to_sql'
    ) as mock_to_sql:

        table = "test_table"

        engine = engine_baseClient._connect()

        engine_baseClient.insert_from_frame(df, table, connection=engine)

        create_engine.assert_called_once_with(
            engine_baseClient.conn_str, connect_args=ANY, echo=ANY
        )

        mock_to_sql.assert_called_once_with(
            table, engine, if_exists='append', index=False,
        )


def test_base_execute_connection_and_params_not_none(engine_baseClient):

    with patch("muttlib.dbconn.base.create_engine") as create_engine:

        engine = engine_baseClient._connect()

        q = "SELECT * FROM {table} WHERE {condition1}"
        params = {"table": "test", "condition1": "id = 1"}

        engine_baseClient.execute(q, params, connection=engine)

        engine.execute.assert_called_once_with(q.format(**params))


@patch("muttlib.dbconn.base.BaseClient.__abstractmethods__", set())
def test_base_to_frame_rise_not_implemented_method():

    with patch("muttlib.dbconn.base.create_engine") as create_engine:

        base_cli = BaseClient(dialect="mysql",)

        q = "SELECT *"

        with pytest.raises(NotImplementedError):
            base_cli.to_frame(q)


@patch("muttlib.dbconn.base.BaseClient.__abstractmethods__", set())
def test_base_insert_from_frame_if_exists_replace_rise_not_implemented_method():

    with patch("muttlib.dbconn.base.create_engine") as create_engine:

        base_cli = BaseClient(dialect="mysql",)

        df = pd.DataFrame({'col1': ['1'], 'col2': ['3.0']})
        table = "test_table"

        with pytest.raises(NotImplementedError):
            base_cli.insert_from_frame(df, table, if_exists='replace')


@patch("muttlib.dbconn.base.BaseClient.__abstractmethods__", set())
def test_base_insert_from_frame_index_true_rise_not_implemented_method():

    with patch("muttlib.dbconn.base.create_engine") as create_engine:

        base_cli = BaseClient(dialect="mysql",)

        df = pd.DataFrame({'col1': ['1'], 'col2': ['3.0']})
        table = "test_table"

        with pytest.raises(NotImplementedError):
            base_cli.insert_from_frame(df, table, index=True)
