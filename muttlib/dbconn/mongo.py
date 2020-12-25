import logging

from pandas.io.json import json_normalize

logger = logging.getLogger(__name__)
from muttlib.dbconn.base import ClientBaseClient

try:
    import pymongo
except ModuleNotFoundError:
    logger.debug("No Mongo support.")

MONGO_DB_TYPE = 'mongo'


class MongoClient(ClientBaseClient):
    """MongoDb client."""

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
        super().__init__(
            username=username,
            password=password,
            database=database,
            host=host,
            port=port,
        )
        if host is None and seeds is None:
            raise ValueError("Either `host` or `seeds` is required and was not found!")

        self.seeds = seeds
        self.replica_set = replica_set
        self.conn_url = self._build_conn_uri()

    def _build_conn_uri(self):
        uri = 'mongodb://'
        if self.username is not None:
            uri += f'{self.username}'
        if self.password is not None:
            uri += f':{self.password}@'
        if self.replica_set is None:
            uri += f'{self.host}:{self.port}'
        else:
            uri += ",".join(self.seeds)
        uri += f'/{self.database}'
        uri += f'?replicaSet={self.replica_set}' if self.replica_set is not None else ''
        return uri

    def _connect(self):
        """Get Database of a MongoClient.

        Returns
        -------
        pymongo.database.Database
        """
        client = pymongo.MongoClient(self.conn_str)
        return client[self.database]

    def _find(self, collection, query=None, fields=None, db=None, limit=0):
        """Execute find operation on collection."""
        if db is None:
            db = self._connect()
        collection = db[collection]
        find_func = getattr(collection, 'find')  # noqa
        res = find_func(query, fields)
        if limit > 0:
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

    def execute(self, collection, action, query=None, **kwargs):
        """Execute query."""
        return NotImplementedError

    def insert_from_frame(self, df, collection, **kwargs):
        """Insert from a Pandas dataframe."""
        return NotImplementedError
