from unittest.mock import patch, ANY, MagicMock

import pandas as pd
import pytest
from itertools import repeat, chain

from muttlib.dbconn import HiveClient


@pytest.fixture
def dummy_db_credentials():
    return {
        "host": "host",
        "port": 10_000,
        "auth": "NOSASL",
        "database": "example",
        "username": "user",
    }


def test_init_without_host_fails(dummy_db_credentials):
    dummy_db_credentials.pop("host")
    with pytest.raises(TypeError):
        HiveClient(**dummy_db_credentials)


def test_init_with_password(dummy_db_credentials):
    dummy_db_credentials["password"] = "password"
    dummy_db_credentials["auth"] = "CUSTOM"
    hive_cli = HiveClient(**dummy_db_credentials)
    assert hive_cli.password == dummy_db_credentials["password"]
    assert hive_cli.auth == dummy_db_credentials["auth"]
    dummy_db_credentials["auth"] = "LDAP"
    hive_cli = HiveClient(**dummy_db_credentials)
    assert hive_cli.password == dummy_db_credentials["password"]
    assert hive_cli.auth == dummy_db_credentials["auth"]


def test_init_with_password_fails(dummy_db_credentials):
    dummy_db_credentials["password"] = "password"
    with pytest.raises(ValueError):
        hive_cli = HiveClient(**dummy_db_credentials)


def test_execute(dummy_db_credentials):
    with patch("pyhive.hive.Connection") as connection:
        hive_cli = HiveClient(**dummy_db_credentials)

        connection.cursor().execute.assert_not_called()
        hive_cli.execute("SELECT *", connection=connection)
        connection.cursor().execute.assert_called_once()


def test_execute_with_incomplete_params_fails(dummy_db_credentials):
    with patch("pyhive.hive.Connection") as connection:
        hive_cli = HiveClient(**dummy_db_credentials)

        q = """
        SELECT * FROM {table}
        WHERE {condition1}
        """
        with pytest.raises(KeyError):
            hive_cli.execute(q, connection=connection, params={"table": "test"})


def test_execute_with_params(dummy_db_credentials):
    with patch("pyhive.hive.Connection") as connection:
        hive_cli = HiveClient(**dummy_db_credentials)

        q = """
        SELECT * FROM {table}
        WHERE {condition1}
        """
        params = {"table": "test", "condition1": "id = 1"}
        hive_cli.execute(q, connection=connection, params=params)
        connection.cursor().execute.assert_called_once_with(
            q.format(**params), async_=ANY
        )


def test_execute_does_not_close_connection(dummy_db_credentials):
    with patch("pyhive.hive.Connection") as connection:
        hive_cli = HiveClient(**dummy_db_credentials)
        hive_cli.execute("SELECT *", connection=connection)
        connection.close.assert_not_called()


def test_execute_closes_connection(dummy_db_credentials):
    with patch("pyhive.hive.Connection") as connection:
        hive_cli = HiveClient(**dummy_db_credentials)
        hive_cli.execute("SELECT *")
        connection().close.assert_called_once()


def test_execute_shows_progress(dummy_db_credentials):
    with patch("pyhive.hive.Connection") as connection, patch(
        "progressbar.ProgressBar"
    ) as progress:
        status = MagicMock()
        # first time return true, then always false
        status.operationState.__eq__.side_effect = chain([True], repeat(False))
        cursor = MagicMock()
        cursor.poll.return_value = status
        connection.return_value.cursor.return_value = cursor
        hive_cli = HiveClient(**dummy_db_credentials)
        hive_cli.execute("SELECT *")
        progress.assert_called_once()
        progress.return_value.update.assert_called_once()
        progress.return_value.finish.assert_called_once()


def test_execute_does_not_show_progress(dummy_db_credentials):
    with patch("pyhive.hive.Connection") as connection, patch(
        "progressbar.ProgressBar"
    ) as progress, patch("pyhive.hive.Cursor") as cursor:
        hive_cli = HiveClient(**dummy_db_credentials)
        hive_cli.execute("SELECT *", show_progress=False)
        progress.assert_not_called()
