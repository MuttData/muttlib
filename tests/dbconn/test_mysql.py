from muttlib.dbconn import MySqlClient


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
