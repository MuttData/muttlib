from unittest.mock import patch, MagicMock, ANY

import pandas as pd

from muttlib.dbconn.base import EngineBaseClient
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
