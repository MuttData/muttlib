from unittest import TestCase

import pytest

from muttlib.dbconn import (
    MySqlClient,
    PgClient,
    SqlServerClient,
    parse_connection_string,
)

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
    assert str(engine.url) == 'postgresql://username:password@host:5555/database'

    client.database = 'test_db'
    client._engine = None
    engine = client.get_engine()
    assert str(engine.url) == 'postgresql://username:password@host:5555/test_db'

    client.driver = 'psycopg2'
    client._engine = None
    engine = client.get_engine()
    assert (
        str(engine.url) == 'postgresql+psycopg2-binary://username:password@host:5555/test_db'
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
    assert str(engine.url) == 'mysql+pymysql://username:password@host:5544/database'


def test_SqlServerClient():
    try:
        client = SqlServerClient(
            host="host", password="password", port=5544, username="username",
        )
    except ValueError as e:
        assert str(e) == 'Database argument is not optional!'


@pytest.mark.parametrize(
    "test_input,expected_output",
    [
        (
            "postgresql+psycopg2://username:password@host:5555/database",
            {
                "database": "database",
                "db_type": "postgres",
                "dialect": "postgresql",
                "driver": "psycopg2",
                "host": "host",
                "password": "password",
                "port": 5555,
                "username": "username",
            },
        ),
        (
            "postgresql+psycopg2://username:password@host/",
            {
                "db_type": "postgres",
                "dialect": "postgresql",
                "driver": "psycopg2",
                "host": "host",
                "password": "password",
                "username": "username",
            },
        ),
        ("postgresql:///", {"db_type": "postgres", "dialect": "postgresql",}),
        ("postgresql://", {"db_type": "postgres", "dialect": "postgresql"}),
        (
            "postgresql://localhost",
            {"db_type": "postgres", "dialect": "postgresql", "host": "localhost"},
        ),
        (
            "postgresql://localhost:5433",
            {
                "db_type": "postgres",
                "dialect": "postgresql",
                "host": "localhost",
                "port": 5433,
            },
        ),
        (
            "postgresql://localhost/mydb",
            {
                "database": "mydb",
                "db_type": "postgres",
                "dialect": "postgresql",
                "host": "localhost",
            },
        ),
        (
            "postgresql://user@localhost",
            {
                "db_type": "postgres",
                "dialect": "postgresql",
                "username": "user",
                "host": "localhost",
            },
        ),
        (
            "postgresql://user:secret@localhost",
            {
                "db_type": "postgres",
                "dialect": "postgresql",
                "host": "localhost",
                "password": "secret",
                "username": "user",
            },
        ),
        (
            "postgresql://other@localhost/otherdb?connect_timeout=10&application_name=myapp",
            {
                "db_type": "postgres",
                "dialect": "postgresql",
                "database": "otherdb",
                "username": "other",
                "host": "localhost",
            },
        ),
        (
            "postgresql://other@localhost/otherdb?connect_timeout=10&application_name=myapp",
            {
                "db_type": "postgres",
                "dialect": "postgresql",
                "database": "otherdb",
                "username": "other",
                "host": "localhost",
            },
        ),
        (
            "mssql+pyodbc://scott:tiger^5HHH@mssql2017:1433/test?driver=ODBC+Driver+13+for+SQL+Server",
            {
                "db_type": "sql_server",
                "dialect": "mssql",
                "driver": "pyodbc",
                "database": "test",
                "username": "scott",
                "password": "tiger^5HHH",
                "host": "mssql2017",
                "port": 1433,
            },
        ),
        (
            "oracle://scott:tiger@127.0.0.1:1521",
            {
                "db_type": "oracle",
                "dialect": "oracle",
                "username": "scott",
                "password": "tiger",
                "host": "127.0.0.1",
                "port": 1521,
            },
        ),
    ],
)
def test_parse_connection_string(test_input, expected_output):
    output = parse_connection_string(test_input)
    tc = TestCase()
    tc.assertDictEqual(output, expected_output)
