from muttlib.dbconn import SqlServerClient


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
