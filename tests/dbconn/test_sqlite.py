from unittest.mock import patch

import pytest

from muttlib.dbconn import SqliteClient


@pytest.fixture()
def sqlite_client():
    return SqliteClient(database="database.db")


def test_url_sqlite():
    client = SqliteClient(
        database="database.db",
    )
    engine = client.get_engine()
    assert str(engine.url) == "sqlite:///database.db"


def test_sqlite_execute():
    with patch("muttlib.dbconn.SqliteClient") as sqlite_cli:
        q = "SELECT * FROM {table} WHERE {condition1}"
        params = {"table": "test", "condition1": "id = 1"}

        sqlite_cli.execute(q, params=params, connection=None)
        sqlite_cli.execute.assert_called_once_with(q, params=params, connection=None)
