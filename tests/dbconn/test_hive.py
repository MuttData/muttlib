from unittest.mock import patch
from unittest.mock import ANY

import pandas as pd
import pytest

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
