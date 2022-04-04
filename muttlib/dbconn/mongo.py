import logging

from pandas import json_normalize

logger = logging.getLogger(__name__)


try:
    import pymongo
except ModuleNotFoundError:
    logger.debug("No Mongo support.")

MONGO_DB_TYPE = 'mongo'


class MongoClient:
    """MongoDb client.

    Parameters
    ----------
    username : str, optional
        user to authenticate
    password : str, optional
        password to authenticate
    database : str, optional
        Default database
    host : str, optional
        Host name of the mongodb instance.
    port: int, optional
        Port number of the mongodb instance.
    seeds: list of str, optional
        DNS-constructed seed list: https://docs.mongodb.com/manual/reference/glossary/#term-seed-list
    replica_set: str,optional
        A cluster of MongoDB servers that implements replication and automated failover.
    """

    def __init__(
        self,
        username=None,
        password=None,
        database=None,
        host=None,
        port=None,
        seeds=None,
        replica_set=None,
    ):
        if host is None and seeds is None:
            raise ValueError("Either `host` or `seeds` is required and was not found!")
        self.username = username
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self.seeds = seeds
        self.replica_set = replica_set

    def _build_conn_uri(self):
        uri = 'mongodb://'
        if self.username is not None:
            uri += f'{self.username}{":"+self.password if self.password else ""}@'
        if self.seeds is None:
            uri += f'{self.host}{":"+self.port if self.port else ""}'
        else:
            uri += ",".join(self.seeds)
        uri += f'/{self.database}'
        uri += f'?replicaSet={self.replica_set}' if self.replica_set is not None else ''
        return uri

    def _connect(self, custom_uri=None):
        if custom_uri is None:
            conn_uri = self._build_conn_uri()
        client = pymongo.MongoClient(conn_uri)
        return client[self.database]

    def _find(self, collection, query=None, fields=None, db=None, limit=0):
        if db is None:
            db = self._connect()
        collection = db[collection]
        find_func = getattr(collection, 'find')  # noqa
        res = find_func(query, fields)
        if (type(limit) == int) and (limit > 0):
            res = res.limit(limit)
        return list(res)

    def to_frame(self, collection, query=None, fields=None, limit=0, no_id=False):
        """Return query result as a Pandas dataframe."""
        cursor = self._find(collection, query=query, fields=fields, limit=limit)
        df = json_normalize(cursor)
        if no_id and not df.empty:
            del df['_id']
        return df

    def insert(self, collection, query=None, db=None):
        """Insert query into collection."""
        if db is None:
            db = self._connect()
        collection = db[collection]
        insert_id = collection.insert_one(query).inserted_id
        return insert_id
