from muttlib.dbconn.base import EngineBaseClient

SQLSERVER_DB_TYPE = 'sql_server'
SQLSERVER_DIALECT = "mssql"


class SqlServerClient(EngineBaseClient):
    """SQLServer client."""

    default_dialect = SQLSERVER_DIALECT
    default_driver = 'pymssql'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.database is None:
            raise ValueError("Database argument is not optional!")
