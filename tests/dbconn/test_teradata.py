from unittest.mock import patch

import pandas as pd
import pytest

from muttlib.dbconn import TeradataClient


@pytest.fixture
def dummy_db_credentials():
    return {
        "host": "192.168.0.1",
        "username": "user12",
        "password": "dummypass",
        "database": "db",
        "table": "teradactyl",
        "authentication": "LDAP",
    }


@pytest.fixture
def df():
    return pd.DataFrame()


def test_teradata_close(dummy_db_credentials):
    with patch("teradatasql.TeradataConnection") as connection, patch(
        "teradatasql.TeradataConnection.close"
    ) as td_close:
        td_cli = TeradataClient(**dummy_db_credentials)
        td_cli.credentials = True
        td_cli.connection = connection
        td_cli.execute("SELECT *")

        assert td_close.call_count == 0
        td_cli.close()
        assert td_close.call_count == 1


def test_execute(dummy_db_credentials):
    with patch("teradatasql.TeradataConnection") as connection:
        td_cli = TeradataClient(**dummy_db_credentials)
        td_cli.connection = connection

        assert connection.cursor.return_value.execute.call_count == 0
        td_cli.execute("SELECT *")
        assert connection.cursor.return_value.execute.call_count == 1


def test_to_frame(dummy_db_credentials):
    with patch("teradatasql.TeradataConnection") as connection:
        td_cli = TeradataClient(**dummy_db_credentials)
        td_cli.connection = connection

        assert connection.cursor.return_value.execute.call_count == 0
        _ = td_cli.to_frame("SELECT *")
        assert connection.cursor.return_value.execute.call_count == 1


def test_insert_from_frame(dummy_db_credentials, df):
    with patch("teradatasql.TeradataConnection") as connection:
        td_cli = TeradataClient(**dummy_db_credentials)
        td_cli.connection = connection

        assert connection.cursor.return_value.execute.call_count == 0
        td_cli.insert_from_frame(df)
        assert connection.cursor.return_value.execute.call_count == 1
