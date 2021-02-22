from unittest.mock import patch, MagicMock

import pytest

from muttlib.dbconn import MongoClient


def test_init_without_host_or_seed_fails():
    with pytest.raises(ValueError, match=r".*[host|seed].*[host|seed].*"):
        MongoClient()


def test_to_frame_uri_with_host_username_password_and_database():
    with patch("pymongo.MongoClient") as cli_mock:
        params = {
            "username": "mutt",
            "password": "data",
            "host": "example_host",
            "database": "example_db",
        }
        mongo_cli = MongoClient(**params)
        collection = MagicMock()
        mongo_cli.to_frame(collection)
        cli_mock.assert_called_once_with(
            f'mongodb://{params["username"]}:{params["password"]}@{params["host"]}/{params["database"]}'
        )
        cli_mock.return_value.__getitem__.assert_called_once_with(params["database"])


def test_to_frame_uri_with_host_username_and_database():
    with patch("pymongo.MongoClient") as cli_mock:
        params = {
            "username": "mutt",
            "host": "example_host",
            "database": "example_db",
        }
        mongo_cli = MongoClient(**params)
        collection = MagicMock()
        mongo_cli.to_frame(collection)
        cli_mock.assert_called_once_with(
            f'mongodb://{params["username"]}@{params["host"]}/{params["database"]}'
        )
        cli_mock.return_value.__getitem__.assert_called_once_with(params["database"])


def test_to_frame_uri_with_seeds():
    with patch("pymongo.MongoClient") as cli_mock:
        params = {
            "seeds": ["db1.example.net:27017", "db2.example.net:2500"],
            "database": "example_db",
        }
        mongo_cli = MongoClient(**params)
        collection = MagicMock()
        mongo_cli.to_frame(collection)
        cli_mock.assert_called_once_with(
            f'mongodb://{",".join(params["seeds"])}/{params["database"]}'
        )
        cli_mock.return_value.__getitem__.assert_called_once_with(params["database"])


def test_to_frame_uri_with_seeds_and_replica_set():
    with patch("pymongo.MongoClient") as cli_mock:
        params = {
            "seeds": ["db1.example.net:27017", "db2.example.net:2500"],
            "database": "example_db",
            "replica_set": "set",
        }
        mongo_cli = MongoClient(**params)
        collection = MagicMock()
        mongo_cli.to_frame(collection)
        cli_mock.assert_called_once_with(
            f'mongodb://{",".join(params["seeds"])}/{params["database"]}?replicaSet={params["replica_set"]}'
        )
        cli_mock.return_value.__getitem__.assert_called_once_with(params["database"])


def test_to_frame_with_query_fields_and_limit():
    with patch("pymongo.MongoClient") as cli_mock:
        params = {
            "username": "mutt",
            "password": "data",
            "host": "example_host",
            "database": "example_db",
        }
        query = {"_id": 123}
        fields = ["field1", "field2"]
        limit = 1
        mongo_cli = MongoClient(**params)
        collection = MagicMock()
        mongo_cli.to_frame(collection, query=query, fields=fields, limit=limit)
        cli_mock.return_value.__getitem__.assert_called_once_with(params["database"])
        db = cli_mock.return_value.__getitem__.return_value
        db.__getitem__.assert_called_once_with(collection)
        db.__getitem__.return_value.find.assert_called_once_with(query, fields)
        db.__getitem__.return_value.find.return_value.limit.assert_called_once_with(
            limit
        )


def test_to_frame_with_negative_limit():
    with patch("pymongo.MongoClient") as cli_mock:
        params = {
            "username": "mutt",
            "password": "data",
            "host": "example_host",
            "database": "example_db",
        }
        limit = -1
        mongo_cli = MongoClient(**params)
        collection = MagicMock()
        mongo_cli.to_frame(collection, limit=limit)
        db = cli_mock.return_value.__getitem__.return_value
        db.__getitem__.return_value.find.return_value.limit.assert_not_called()
