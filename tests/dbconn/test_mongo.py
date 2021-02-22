from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from muttlib.dbconn import MongoClient


@pytest.fixture
def dummy_params():
    return {
        "username": "mutt",
        "password": "data",
        "host": "example_host",
        "database": "example_db",
    }


def test_init_without_host_or_seed_fails():
    with pytest.raises(ValueError, match=r".*[host|seed].*[host|seed].*"):
        MongoClient()


def test_to_frame_uri_with_host_username_password_and_database(dummy_params):
    with patch("pymongo.MongoClient") as cli_mock:
        mongo_cli = MongoClient(**dummy_params)
        collection = MagicMock()
        mongo_cli.to_frame(collection)
        cli_mock.assert_called_once_with(
            f'mongodb://{dummy_params["username"]}:{dummy_params["password"]}@{dummy_params["host"]}/{dummy_params["database"]}'
        )
        cli_mock.return_value.__getitem__.assert_called_once_with(
            dummy_params["database"]
        )


def test_to_frame_uri_with_host_username_and_database(dummy_params):
    with patch("pymongo.MongoClient") as cli_mock:
        dummy_params.pop("password")
        mongo_cli = MongoClient(**dummy_params)
        collection = MagicMock()
        mongo_cli.to_frame(collection)
        cli_mock.assert_called_once_with(
            f'mongodb://{dummy_params["username"]}@{dummy_params["host"]}/{dummy_params["database"]}'
        )
        cli_mock.return_value.__getitem__.assert_called_once_with(
            dummy_params["database"]
        )


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


def test_to_frame_with_query_fields_and_limit(dummy_params):
    with patch("pymongo.MongoClient") as cli_mock:
        query = {"_id": 123}
        fields = ["field1", "field2"]
        limit = 1
        mongo_cli = MongoClient(**dummy_params)
        collection = MagicMock()
        mongo_cli.to_frame(collection, query=query, fields=fields, limit=limit)
        cli_mock.return_value.__getitem__.assert_called_once_with(
            dummy_params["database"]
        )
        db = cli_mock.return_value.__getitem__.return_value
        db.__getitem__.assert_called_once_with(collection)
        db.__getitem__.return_value.find.assert_called_once_with(query, fields)
        db.__getitem__.return_value.find.return_value.limit.assert_called_once_with(
            limit
        )


def test_to_frame_with_negative_limit(dummy_params):
    with patch("pymongo.MongoClient") as cli_mock:
        limit = -1
        mongo_cli = MongoClient(**dummy_params)
        collection = MagicMock()
        mongo_cli.to_frame(collection, limit=limit)
        db = cli_mock.return_value.__getitem__.return_value
        db.__getitem__.return_value.find.return_value.limit.assert_not_called()


def test_to_frame_with_invalid_limit(dummy_params):
    with patch("pymongo.MongoClient") as cli_mock:
        limit = "not a number"
        mongo_cli = MongoClient(**dummy_params)
        collection = MagicMock()
        mongo_cli.to_frame(collection, limit=limit)
        db = cli_mock.return_value.__getitem__.return_value
        db.__getitem__.return_value.find.return_value.limit.assert_not_called()


def test_to_frame_no_id(dummy_params):
    with patch("pymongo.MongoClient") as cli_mock, patch(
        "muttlib.dbconn.mongo.json_normalize"
    ) as json_normalize:
        df = pd.DataFrame(data=[[1, 2]], columns=["_id", "ej_col"])
        json_normalize.return_value = df
        mongo_cli = MongoClient(**dummy_params)
        collection = MagicMock()
        mongo_cli.to_frame(collection, no_id=True)
        assert df.columns == ["ej_col"]


def test_insert(dummy_params):
    with patch("pymongo.MongoClient") as cli_mock:
        mongo_cli = MongoClient(**dummy_params)
        query = {"field1": "value1", "field2": "value2"}
        collection = MagicMock()
        example_id = 4321
        id_mock = MagicMock(inserted_id=example_id)
        cli_mock.return_value.__getitem__.return_value.__getitem__.return_value.insert_one.return_value = (
            id_mock
        )
        ret_value = mongo_cli.insert(collection, query=query)
        cli_mock.return_value.__getitem__.return_value.__getitem__.return_value.insert_one.assert_called_once_with(
            query
        )
        assert ret_value == example_id
