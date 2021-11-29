from itertools import chain, repeat
from unittest.mock import ANY, MagicMock, call, patch

import pytest
import pandas as pd

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
        cursor = MagicMock()
        connection.cursor.return_value = cursor

        hive_cli = HiveClient(**dummy_db_credentials)
        q = "SELECT *"

        cursor.execute.assert_not_called()
        hive_cli.execute(q, connection=connection, async_=False)
        connection.cursor.assert_called_once()
        cursor.execute.assert_called_once_with(q, async_=False)


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
        hive_cli.execute(q, connection=connection, params=params, async_=False)
        connection.cursor.assert_called_once()
        connection.cursor.return_value.execute.assert_called_once_with(
            q.format(**params), async_=False
        )


def test_execute_does_not_close_connection(dummy_db_credentials):
    with patch("pyhive.hive.Connection") as connection:
        cursor = MagicMock()
        connection.cursor.return_value = cursor

        hive_cli = HiveClient(**dummy_db_credentials)
        hive_cli.execute("SELECT *", connection=connection)
        connection.cursor.assert_called_once()
        connection.close.assert_not_called()
        cursor.execute.assert_called_once()


def test_execute_closes_connection(dummy_db_credentials):
    with patch("pyhive.hive.Connection") as connection:
        cursor = MagicMock()
        connection.return_value.cursor.return_value = cursor

        hive_cli = HiveClient(**dummy_db_credentials)
        hive_cli.execute("SELECT *")
        connection.return_value.cursor.assert_called_once()
        connection.return_value.close.assert_called_once()
        cursor.execute.assert_called_once()


def test_execute_shows_progress(dummy_db_credentials):
    with patch("pyhive.hive.Connection") as connection, patch(
        "progressbar.ProgressBar"
    ) as progress:
        iterations = 4
        status = MagicMock()
        # first time return true, then always false
        status.operationState.__eq__.side_effect = chain(
            [True] * iterations, repeat(False)
        )
        status.progressUpdateResponse = None
        cursor = MagicMock()
        cursor.poll.return_value = status
        connection.return_value.cursor.return_value = cursor
        cursor.fetch_logs.side_effect = [
            [f"text ({i}/{iterations*10}) more_text"]
            for i in range(10, (iterations + 1) * 10, 10)
        ]

        hive_cli = HiveClient(**dummy_db_credentials)
        hive_cli.execute("SELECT *")
        progress.assert_called_once()
        assert progress.return_value.update.call_count == iterations
        # this seems to be hardcoded in HiveClient
        max_value = 100
        progress.return_value.update.assert_has_calls(
            [
                call(v * max_value)
                for v in [i / iterations for i in range(1, iterations + 1)]
            ]
        )
        progress.return_value.finish.assert_called_once()


def test_execute_does_not_show_progress(dummy_db_credentials):
    with patch("pyhive.hive.Connection") as connection, patch(
        "progressbar.ProgressBar"
    ) as progress:
        hive_cli = HiveClient(**dummy_db_credentials)
        hive_cli.execute("SELECT *", show_progress=False)
        progress.assert_not_called()
