from muttlib.dbconn import TrinoClient


def test_TrinoClient():
    client = TrinoClient(
        database="database",
        host="host",
        password="password",
        port=443,
        username="username",
    )
    engine = client.get_engine()
    assert str(engine.url) == "trino://username:password@host:443/database"

    client.database = "test_db"
    client._engine = None
    engine = client.get_engine()
    assert str(engine.url) == "trino://username:password@host:443/test_db"
