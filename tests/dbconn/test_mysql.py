from unittest.mock import ANY, MagicMock, patch

from muttlib.dbconn import MySqlClient


def test_url():
    client = MySqlClient(
        database="database",
        host="host",
        password="password",
        port=5544,
        username="username",
    )
    engine = client.get_engine()
    assert str(engine.url) == "mysql+pymysql://username:password@host:5544/database"


def test_insert_from_frame_connects():
    with patch("muttlib.dbconn.base.create_engine") as create_engine:
        client = MySqlClient(
            database="database",
            host="host",
            password="password",
            port=5544,
            username="username",
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
            if_exists="append",
            index=False,
        )
