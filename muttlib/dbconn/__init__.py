from sqlalchemy.engine.url import make_url

from muttlib.dbconn.base import BaseClient
from muttlib.dbconn.bigquery import BIGQUERY_DB_TYPE, BigQueryClient
from muttlib.dbconn.hive import HIVE_DB_TYPE, HiveClient
from muttlib.dbconn.ibis import IBIS_DB_TYPE, IbisClient
from muttlib.dbconn.mongo import MONGO_DB_TYPE, MongoClient
from muttlib.dbconn.mysql import MYSQL_DB_TYPE, MySqlClient
from muttlib.dbconn.oracle import ORACLE_DB_TYPE, OracleClient
from muttlib.dbconn.postgres import POSTGRES_DB_TYPE, POSTGRES_DIALECT, PgClient
from muttlib.dbconn.sqlite import SQLITE_DB_TYPE, SqliteClient
from muttlib.dbconn.sqlserver import (
    SQLSERVER_DB_TYPE,
    SQLSERVER_DIALECT,
    SqlServerClient,
)
from muttlib.dbconn.teradata import TERADATA_DB_TYPE, TeradataClient

connectors = {
    BIGQUERY_DB_TYPE: BigQueryClient,
    HIVE_DB_TYPE: HiveClient,
    IBIS_DB_TYPE: IbisClient,
    MONGO_DB_TYPE: MongoClient,
    MYSQL_DB_TYPE: MySqlClient,
    ORACLE_DB_TYPE: OracleClient,
    POSTGRES_DB_TYPE: PgClient,
    SQLITE_DB_TYPE: SqliteClient,
    SQLSERVER_DB_TYPE: SqlServerClient,
    TERADATA_DB_TYPE: TeradataClient,
}


def get_client(creds):
    """Get a db client from a credentials dict."""
    db_type = creds.pop('db_type')
    return db_type, connectors[db_type](**creds)


def parse_connection_string(connstr):
    """Parse a connection string and return the arguments to call muttlib.dbconn.get_client

    This should work at least work on most of Postgres connection URIs.
    Ref: https://www.postgresql.org/docs/10/libpq-connect.html

    Note that connection specific args after '?' are ignored.

    Args:
        connstr (str): URI-like database connection string.

    Return:
        dict: Parsed components of the connection string.
    """
    r = make_url(connstr)
    dialect = r.get_backend_name()
    db_type = {
        POSTGRES_DIALECT: POSTGRES_DB_TYPE,
        SQLSERVER_DIALECT: SQLSERVER_DB_TYPE,
    }.get(dialect, dialect)
    rv = {
        "db_type": db_type,
        "dialect": dialect,
    }
    if r.database:
        rv["database"] = r.database

    if "+" in r.drivername:
        rv["driver"] = r.drivername.split("+")[1]

    if r.username:
        rv["username"] = r.username
    if r.password:
        rv["password"] = r.password

    if r.host:
        rv["host"] = r.host
    if r.port:
        rv["port"] = r.port

    return rv


def get_client_from_connstr(connstr):
    args = parse_connection_string(connstr)
    rv = get_client(args)
    return rv
