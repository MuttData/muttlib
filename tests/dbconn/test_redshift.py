from muttlib.dbconn import RedshiftClient


def test_RedshiftClient():
    client = RedshiftClient(
        database="database",
        host="host",
        password="password",
        port=5439,
        username="username",
    )
    engine = client.get_engine()
    assert str(engine.url) == 'redshift://username:password@host:5439/database'

    client.database = 'test_db'
    client._engine = None
    engine = client.get_engine()
    assert str(engine.url) == 'redshift://username:password@host:5439/test_db'
