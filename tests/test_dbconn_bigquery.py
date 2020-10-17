from unittest.mock import patch

import pandas as pd
import pytest

from muttlib.dbconn import BigQueryClient


@pytest.fixture
def dummy_db_credentials():
    return {
        "db": "192.168.0.1",
        "table": "teradactyl",
        "auth": "{'test':'test1'}",
        "auth_file": None,
        "project": "project",
    }


@pytest.fixture
def df():
    return pd.DataFrame()


def test_bigquery_close(dummy_db_credentials):
    with patch("google.cloud.bigquery.client.Client") as client, patch(
        "google.cloud.bigquery.client.Client.close"
    ) as bq_close:
        bq_cli = BigQueryClient(**dummy_db_credentials)
        bq_cli.credentials = True
        bq_cli.client = client
        bq_cli.execute("SELECT *")

        assert bq_close.call_count == 0
        bq_cli.close()
        assert bq_close.call_count == 1


def test_execute(dummy_db_credentials):
    with patch("google.cloud.bigquery.client.Client") as client, patch(
        "google.cloud.bigquery.client.Client.query"
    ) as bq_query:
        bq_cli = BigQueryClient(**dummy_db_credentials)
        bq_cli.credentials = True
        bq_cli.client = client

        assert bq_query.call_count == 0
        bq_cli.execute("SELECT *")
        assert bq_query.call_count == 1


def test_to_frame(dummy_db_credentials):
    with patch("google.cloud.bigquery.client.Client") as client, patch(
        "google.cloud.bigquery.client.Client.query"
    ) as bq_query:

        bq_cli = BigQueryClient(**dummy_db_credentials)
        bq_cli.credentials = True
        bq_cli.client = client

        assert bq_query.call_count == 0
        _ = bq_cli.to_frame("SELECT *")
        assert bq_query.call_count == 1


def test_insert_from_frame(dummy_db_credentials, df):
    with patch("google.cloud.bigquery.client.Client") as client, patch(
        "google.cloud.bigquery.client.Client.load_table_from_dataframe"
    ) as bq_load:
        bq_cli = BigQueryClient(**dummy_db_credentials)
        bq_cli.credentials = True
        bq_cli.client = client

        assert bq_load.call_count == 0
        bq_cli.insert_from_frame(df, create_first=False)
        assert bq_load.call_count == 1
