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
