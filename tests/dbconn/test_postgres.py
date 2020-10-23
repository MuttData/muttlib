from muttlib.dbconn import PgClient


def test_PgClient():
    client = PgClient(
        database="database",
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
        str(engine.url) == 'postgresql+psycopg2://username:password@host:5555/test_db'
    )
