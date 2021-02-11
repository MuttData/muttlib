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


def test_execute_with_no_client():
    with patch("ibis.impala") as impala:
        ibis_cli = IbisClient("host")
        q = "SELECT *"
        ibis_cli.execute_new(q, return_cursor=False)
        impala.connect.assert_called_once()
        impala.connect.return_value.raw_sql.assert_called_once_with(q, results=False)


def test_execute_with_client():
    with patch("ibis.impala") as impala:
        ibis_cli = IbisClient("host")
        client = MagicMock()
        q = "SELECT *"
        ibis_cli.execute_new(q, client=client, return_cursor=False)
        impala.connect.assert_not_called()
        client.raw_sql.assert_called_once_with(q, results=False)


def test_execute_returns_cursor():
    with patch("ibis.impala") as impala:
        ibis_cli = IbisClient("host")
        client = MagicMock()
        q = "SELECT *"
        ret = ibis_cli.execute_new(q, client=client, return_cursor=True)
        client.raw_sql.assert_called_once_with(q, results=True)


def test_to_frame_via_hdfs_and_no_hdfs_client_fails():
    with patch("ibis.impala") as impala:
        ibis_cli = IbisClient("host")
        q = "SELECT *"
        with pytest.raises(ValueError, match=r".*hdfs\Wclient.*"):
            ibis_cli.to_frame(q, via_hdfs=True)


def test_to_frame_via_hdfs_and_no_cache_dir_fails():
    with patch("ibis.impala") as impala:
        ibis_cli = IbisClient("host", hdfs_host="", hdfs_port="", hdfs_username="",)
        q = "SELECT *"
        with pytest.raises(ValueError, match=r".*dir.*"):
            ibis_cli.to_frame(q, via_hdfs=True)


def test_to_frame_via_hdfs_and_refresh_cache():
    with patch("ibis.impala") as impala, patch("shutil.rmtree") as rm, patch(
        "pyarrow.parquet.read_table"
    ) as pq:
        ibis_cli = IbisClient("host", hdfs_host="", hdfs_port="", hdfs_username="")
        q = "SELECT *"
        ibis_cli.to_frame(q, via_hdfs=True, cache_dir=MagicMock(), refresh_cache=True)
        rm.assert_called_once()


# def test_to_frame_creates_tmp_table():
#     with patch("ibis.impala") as impala:
#         local_tmp_table_dir = MagicMock()
#         local_tmp_table_dir.exists.return_value = False
#         cache_dir = MagicMock()
#         cache_dir.__truediv__.return_value = local_tmp_table_dir
#         ibis_cli = IbisClient("host", hdfs_host="", hdfs_port="", hdfs_username="")
#         q = "SELECT *"
#         ibis_cli.to_frame(q, via_hdfs=True, cache_dir=cache_dir, refresh_cache=True)
#         assert False
