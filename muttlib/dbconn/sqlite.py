from muttlib.dbconn.base import EngineBaseClient

SQLITE_DB_TYPE = 'sqlite'


class SqliteClient(EngineBaseClient):
    """Create SQLite DB client."""

    default_dialect = 'sqlite'
