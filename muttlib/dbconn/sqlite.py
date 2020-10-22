from muttlib.dbconn.base import BaseClient

SQLITE_DB_TYPE = 'sqlite'


class SqliteClient(BaseClient):
    """Create SQLite DB client."""

    default_dialect = 'sqlite'
