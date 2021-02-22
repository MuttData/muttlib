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
