from unittest.mock import patch, MagicMock, ANY

from muttlib.dbconn import OracleClient
import pytest


@pytest.fixture(scope='session')
def oracleClient():
    client = OracleClient(
        username="scott",
        host="127.0.0.1",
        port=1521,
        dialect="oracle",
        password="tiger",
    )

    return client


@pytest.fixture(scope='session')
def oracleClient_withScheme():
    client = OracleClient(
        username="scott",
        host="127.0.0.1",
        port=1521,
        dialect="oracle",
        password="tiger",
        schema='test',
    )

    return client


def url_connection(oracleClient):
    engine = oracleClient.get_engine()
    assert str(engine.url) == 'oracle://scott:tiger@127.0.0.1:1521'


def url_connection_with_scheme(oracleClient_withScheme):
    engine = oracleClient_withScheme.get_engine()
    assert str(engine.url) == 'oracle://scott:tiger@127.0.0.1:1521/test'


def url_connection_with_scheme_error(oracleClient):
    engine = oracleClient.get_engine()
    assert engine.url != 'oracle://scott:tiger@127.0.0.1:1521/test'


def test_oracle_insert_from_frame_connects(oracleClient):

    with patch("muttlib.dbconn.base.create_engine") as create_engine:
        client = OracleClient(
            username="scott",
            host="127.0.0.1",
            port=1521,
            dialect="oracle",
            password="tiger",
        )
        df = MagicMock()
        table = "test_table"
        client.insert_from_frame(df, table)
        create_engine.assert_called_once_with(
            client.conn_str, connect_args=ANY, echo=ANY
        )
        df.to_sql.assert_called_once_with(
            table,
            create_engine.return_value.connect.return_value.__enter__.return_value,
            if_exists='append',
            index=False,
        )
