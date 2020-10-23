from contextlib import closing
import logging

import pandas as pd

from muttlib.dbconn.base import BaseClient
import muttlib.utils as utils

logger = logging.getLogger(__name__)
try:
    import teradatasql
except ModuleNotFoundError:
    logger.debug("No Teradata support.")


TERADATA_DB_TYPE = 'teradata'


class TeradataClient(BaseClient):
    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        database: str,
        table: str,
        port=1025,
        authentication="LDAP",
    ):  # pylint: disable-msg=too-many-arguments
        """Wrapper around Teradata Connection.
        host : str
            Host name of Teradata.
        username : str
            LDAP user to authenticate
        password : str
            password to authenticate
        database: str
            Database name where data will be inserted.
        table: str
            Table name where data will be inserted.
        port : int
            Teradata's port
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.authentication = authentication
        self.connection = None
        self.database = database
        self.table = table

    def _connect(self):
        if self.connection is None:
            self.connection = teradatasql.connect(
                host=self.host,
                dbs_port=self.port,
                user=self.username,
                password=self.password,
                logmech=self.authentication,
            )
        return self.connection

    def close(self):
        if self.connection is not None:
            self.connection.close()

    def execute(self, sql, params=None, connection=None):  # pylint: disable=W0613
        """Execute sql statement."""
        if connection is None:
            connection = self._connect()
        cursor = connection.cursor()
        sql = utils.path_or_string(sql)
        if params is not None:
            sql = sql.format(**params)
        logger.info("executing query:\n%s", sql)
        return cursor.execute(sql)

    def fetch_all(self, sql, params=None, connection=None):
        """Return all tuples."""
        with closing(self.execute(sql, params=params, connection=connection)) as cursor:
            return cursor.fetchall()

    def to_frame(self, sql, params=None, connection=None):
        """Return sql execution as Pandas dataframe."""
        with closing(self.execute(sql, params=params, connection=connection)) as cursor:
            data = cursor.fetchall()
            if data:
                df = pd.DataFrame(data, columns=[c[0] for c in cursor.description])
            else:
                df = pd.DataFrame()
            return df

    def insert_from_frame(
        self, df, create_first=True, create_sql=None,
    ):
        """Insert from a Pandas dataframe.

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe with data to be inserted.
        create_first: bool
            Whether to create table before attempting insertion.
        create_sql: str or path
            SQL query string or path to file with create statements.
        """
        full_table = f'{self.database}.{self.table}'
        logger.info("Going to insert data into %s...", full_table)

        if create_first:
            self._create_table(self.database, self.table, create_sql)

        connection = self._connect()
        with closing(connection.cursor()) as cursor:
            cols_placeholder = ('?, ' * len(df.columns))[:-2]  # remove trailing comma
            # Note: teradata insert doesn't guarantee that row order is kept
            records = tuple(df.values.tolist())
            sql_insert = f'INSERT INTO {full_table} ({cols_placeholder})'
            cursor.execute(sql_insert, records)
        logger.info("Inserted %s records into %s", len(df), full_table)

    def _create_table(self, db, table, sql):
        """Create table if it doesn't exist.

        Parameters
        ----------
        db: str
            Database name where data will be inserted.
        table: str
            Table name where data will be inserted.
        sql: str or path
            SQL query string or path to file with create statements.
        """
        table_exists = self._table_exists(db, table)
        if not table_exists:
            self.execute(sql, params={'db': db, 'table': table})
            logger.info("Created %s.%s", db, table)

    def _table_exists(self, db, table):
        """Check whether table exists in db."""
        logger.info("Checking if %s.%s exists...", db, table)
        sql = f"""
        SELECT * FROM dbc.tables WHERE databasename = '{db}' AND tablename = '{table}'
        """  # nosec
        df = self.to_frame(sql)
        return False if df.empty else True
