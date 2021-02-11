from unittest.mock import patch, MagicMock

import pytest

from muttlib.dbconn import IbisClient


def test_init_without_host_fails():
    with pytest.raises(TypeError):
        IbisClient()


def test_temp_db_changes_config():
    with patch("ibis.config") as config:
        temp_db = MagicMock()
        ibis_cli = IbisClient("host", temp_db=temp_db)
        config.set_option.assert_called_once_with("temp_db", temp_db)


def test_temp_hdfs_path_changes_config():
    with patch("ibis.config") as config:
        temp_hdfs_path = MagicMock()
        ibis_cli = IbisClient("host", temp_hdfs_path=temp_hdfs_path)
        config.set_option.assert_called_once_with("temp_hdfs_path", temp_hdfs_path)


def test_execute_with_no_client_establishes_connection():
    with patch("ibis.impala") as impala:
        ibis_cli = IbisClient("host")
        q = "SELECT *"
        ibis_cli.execute_new(q)
        impala.connect.assert_called_once()


def test_execute_with_client_does_not_establish_connection():
    with patch("ibis.impala") as impala:
        ibis_cli = IbisClient("host")
        client = MagicMock()
        q = "SELECT *"
        ibis_cli.execute_new(q, client=client)
        impala.connect.assert_not_called()
