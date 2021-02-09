from unittest.mock import patch

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
        "password": "password",
    }


def test_execute(dummy_db_credentials):
    with patch("pyhive.hive.Connection") as connection, patch(
        "pyhive.hive.Cursor"
    ) as hive_query:
        hive_cli = HiveClient(**dummy_db_credentials)

        assert hive_query.call_count == 0
        hive_cli.execute("SELECT *", connection=connection)
        assert hive_query.call_count == 1
