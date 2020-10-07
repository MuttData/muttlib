from unittest import TestCase

import pytest

from muttlib.dbconn import PgClient, MySqlClient, SqlServerClient

# pip install psycopg2 pymysql

def test_PgClient():
    client = PgClient(
        database="database",
        # dialect="postgresql",
        # driver="psycopg2",
        host="host",
        password="password",
        port=5555,
        username="username",
    )
    engine = client.get_engine()
    assert (
        str(engine.url) == 'postgresql://username:password@host:5555/database'
    )

    client.database = 'test_db'
    client._engine = None
    engine = client.get_engine()
    assert (
        str(engine.url) == 'postgresql://username:password@host:5555/test_db'
    )

    client.driver = 'psycopg2'
    client._engine = None
    engine = client.get_engine()
    assert (
        str(engine.url) == 'postgresql+psycopg2://username:password@host:5555/test_db'
    )



def test_MySqlClient():
    client = MySqlClient(
        database="database",
        host="host",
        password="password",
        port=5544,
        username="username",
    )
    engine = client.get_engine()
    assert (
        str(engine.url) == 'mysql+pymysql://username:password@host:5544/database'
    )


def test_SqlServerClient():
    try:
        client = SqlServerClient(
            host="host",
            password="password",
            port=5544,
            username="username",
        )
    except ValueError as e:
        assert str(e) == 'Database argument is not optional!'
