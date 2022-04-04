from unittest.mock import patch, create_autospec

import pytest
import pandas as pd

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
    with patch("google.cloud.bigquery.client.Client") as connection, patch(
        "google.cloud.bigquery.client.Client.close"
    ) as bq_close:
        bq_cli = BigQueryClient(**dummy_db_credentials)
        bq_cli.credentials = True
        bq_cli.connection = connection
        bq_cli.execute("SELECT *")

        assert bq_close.call_count == 1
        bq_cli.close()
        assert bq_close.call_count == 2


def test_execute(dummy_db_credentials):
    with patch("google.cloud.bigquery.client.Client") as connection, patch(
        "google.cloud.bigquery.client.Client.query"
    ) as bq_query:
        bq_cli = BigQueryClient(**dummy_db_credentials)
        bq_cli.credentials = True
        bq_cli.connection = connection

        assert bq_query.call_count == 0
        bq_cli.execute("SELECT *")
        assert bq_query.call_count == 1


def test_execute_bigquery_with_incomplete_params_fails(dummy_db_credentials):
    with patch("google.cloud.bigquery.client.Client") as connection, patch(
        "google.cloud.bigquery.client.Client.query"
    ) as bq_query:
        bq_cli = BigQueryClient(**dummy_db_credentials)
        bq_cli.credentials = True
        bq_cli.connection = connection

        q = """
        SELECT * FROM {table}
        WHERE {condition1}
        """
        with pytest.raises(KeyError):
            bq_cli.execute(q, connection=connection, params={"table": "teradactyl"})


def test_execute_bigquery_with_params(dummy_db_credentials):
    with patch("google.cloud.bigquery.client.Client") as connection, patch(
        "google.cloud.bigquery.client.Client.query"
    ) as bq_query:
        bq_cli = BigQueryClient(**dummy_db_credentials)
        bq_cli.credentials = True
        bq_cli.connection = connection

        q = """
        SELECT * FROM {table}
        WHERE {condition1}
        """
        params = {"table": "teradactyl", "condition1": "id = 1"}
        bq_cli.execute(q, connection=connection, params=params)
        connection.query.assert_called_once_with(q.format(**params))


def test_to_frame(dummy_db_credentials):
    with patch("google.cloud.bigquery.client.Client") as connection, patch(
        "google.cloud.bigquery.client.Client.query"
    ) as bq_query:

        bq_cli = BigQueryClient(**dummy_db_credentials)
        bq_cli.credentials = True
        bq_cli.connection = connection

        assert bq_query.call_count == 0
        _ = bq_cli.to_frame("SELECT *")
        assert bq_query.call_count == 1


def test_insert_from_frame(dummy_db_credentials, df):
    with patch("google.cloud.bigquery.client.Client") as connection, patch(
        "google.cloud.bigquery.client.Client.load_table_from_dataframe"
    ) as bq_load:
        bq_cli = BigQueryClient(**dummy_db_credentials)
        bq_cli.credentials = True
        bq_cli.connection = connection

        assert bq_load.call_count == 0
        bq_cli.insert_from_frame(df, create_first=False)
        assert bq_load.call_count == 1


def test_insert_from_frame_create_first_true(dummy_db_credentials):
    with patch("google.cloud.bigquery.client.Client") as connection, patch(
        "google.cloud.bigquery.client.Client.load_table_from_dataframe"
    ) as bq_load:

        df = pd.DataFrame({'col1': [1], 'col2': [3.0]})

        bq_cli = BigQueryClient(**dummy_db_credentials)
        bq_cli.credentials = True
        bq_cli.connection = connection

        bq_cli.insert_from_frame(df, create_first=True)
        bq_load.assert_called_once_with(df, bq_cli.table_id)


def test_connect_credentials_none(dummy_db_credentials):
    with patch("google.cloud.bigquery.client.Client") as connection, patch(
        "google.cloud.bigquery.client.Client.query"
    ) as bq_query, patch(
        "muttlib.dbconn.bigquery.BigQueryClient._read_cred"
    ) as mock_read_cred:
        bq_cli = BigQueryClient(**dummy_db_credentials)
        bq_cli.credentials = None
        bq_cli.connection = connection

        bq_cli.execute("SELECT *")
        mock_read_cred.assert_called_once_with(bq_cli.auth, bq_cli.auth_file)


def test_insert_from_frame_create_first_true_with_connection(dummy_db_credentials, df):
    with patch("google.cloud.bigquery.client.Client") as connection:
        bq_cli = BigQueryClient(**dummy_db_credentials)
        bq_cli.credentials = True

        bq_cli.insert_from_frame(df, create_first=True, connection=connection)
        connection.load_table_from_dataframe.assert_called_once_with(
            df, bq_cli.table_id
        )
        connection.load_table_from_dataframe.return_value.result.assert_called_once()


def test_insert_from_frame_create_first_true_get_table_not_found(
    dummy_db_credentials, df
):
    from google.cloud import exceptions

    bq_cli = BigQueryClient(**dummy_db_credentials)
    bq_cli.credentials = True

    with patch("google.cloud.bigquery.client.Client") as connection, patch.object(
        bq_cli, 'execute'
    ) as execute_mock:

        bq_cli.connection = connection
        connection.get_table.side_effect = exceptions.NotFound('Not Found')

        q = "CREATE TABLE {table_id}"
        params = {'table_id': 'test'}

        bq_cli.insert_from_frame(
            df,
            table=params['table_id'],
            create_first=True,
            create_sql=q.format(**params),
            connection=connection,
        )

        connection.get_table.assert_called_once_with(params['table_id'])

        execute_mock.assert_called_once_with(
            q.format(**params),
            params=params,
            connection=connection,
        )
