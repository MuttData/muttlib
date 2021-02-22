from muttlib.dbconn import OracleClient
import pytest


@pytest.fixture(scope='session')
def oracleClient():
    client = OracleClient(
        username="scott",
        host="127.0.0.1",
        port=1521,
        db_type="oracle",
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
        db_type="oracle",
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
