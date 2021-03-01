from contextlib import closing
import json
import logging
from pathlib import Path
from typing import Optional, Union

from muttlib.dbconn.base import BaseClient
import muttlib.utils as utils

logger = logging.getLogger(__name__)
try:
    from google.cloud import bigquery, exceptions
    from google.oauth2 import service_account
except ModuleNotFoundError:
    logger.debug("No BigQuery support.")


BIGQUERY_DB_TYPE = 'bigquery'


class BigQueryClient(BaseClient):
    """BigQuery Client."""

    def __init__(
        self,
        db: str,
        table: str,
        project: str,
        auth: Optional[str] = None,
        auth_file: Optional[Union[str, Path]] = None,
    ):
        """Wrapper around BigQuery Connection.

        Parameters
        ----------
        db: str
            Database name.
        table: str
            Table/dataset name.
        project: str
            Project name to work with.
        auth: str, Optional
            credentials to create the connection.
        auth_file: str or Path, Optional
            file where credentials are stored.
        """
        self.auth = auth
        self.auth_file = auth_file
        self.credentials = None
        self.project = project
        self.connection = None
        self.db = db
        self.table = table

    @property
    def table_id(self):
        return f"{self.project}.{self.db}.{self.table}"

    def _read_cred(
        self, auth: Optional[str], auth_file: Optional[Union[str, Path]],
    ):
        """Create valid OAuth2 credentials for bigquery.

        Parameters
        ----------
        auth: str, Optional
            credentials to create the connection.
        auth_file: str or Path, Optional
            file where credentials are stored.

        Returns
        ----------
        google.auth.service_account.Credentials: The constructed credentials.
        """
        if auth is None:
            return service_account.Credentials.from_service_account_file(auth_file)
        else:
            return service_account.Credentials.from_service_account_info(auth)

    def _connect(self):
        """Create a BigQuery Client.

        Returns
        ----------
        google.cloud.bigquery.client
        """
        if self.credentials is None:
            self.credentials = self._read_cred(self.auth, self.auth_file)
        if self.connection is None:
            self.connection = bigquery.Client(
                project=self.project, credentials=self.credentials
            )
        return self.connection

    def close(self):
        """Close connection."""
        if self.connection is not None:
            self.connection.close()

    def execute(self, sql, params=None, connection=None):  # pylint: disable=W0613
        """Execute sql statement.

        Parameters
        ----------
        sql : str or path
            SQL query string or path to file with Select statements.
        params : dict, optional
            parameters to add into the SQL query to execute.
        connection : google.cloud.bigquery.client, optional
            client to the database, if it's already created.

        Returns
        -------
        google.cloud.bigquery.job.QueryJob
        """
        sql = utils.path_or_string(sql)
        if params is not None:
            sql = sql.format(**params)
        logger.info(f"Executing query:\n{sql}")

        if connection is not None:
            return connection.query(sql)

        with closing(self._connect()) as connection:
            return connection.query(sql)

    def to_frame(self, sql, params=None, connection=None):  # pylint: disable=W0221
        """Return sql execution as Pandas dataframe.

        Parameters
        ----------
        sql : str or path
            SQL query string or path to file with Select statements.
        params : dict, optional
            parameters to add into the SQL query to execute.
        connection : google.cloud.bigquery.client
            connection to the database, if it's already created.

        Returns
        -------
        pandas.DataFrame
        """
        return self.execute(sql, params, connection).to_dataframe()

    def insert_from_frame(  # pylint: disable=W0221
        self,
        df,
        table=None,
        create_first=True,
        create_sql=None,
        connection=None,
        **kwargs,
    ):
        """Insert from a Pandas dataframe.

        If it is an existing table, the schema of the DataFrame must match the schema
        of the destination table. If the table does not yet exist,
        the schema is inferred from the DataFrame.

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe with data to be inserted.
        table: str, optional.
            table name to insert data, if none use the created.
            table pattern: "<project>.<database>.<table>"
        create_first: bool, optional
            Whether to create table before attempting insertion.
        create_sql: str or path, optional
            SQL query string or path to file with create statements.
        connection: google.cloud.bigquery.client, optional
            client to the database, if it's already created.
        """
        if table is None:
            table = self.table_id

        logger.info(f"Going to insert data into {table}")

        if connection is not None:
            if create_first:
                self._create_table(table, create_sql, connection)

            job = connection.load_table_from_dataframe(df, table)
            job.result()  # Wait for the job to complete.

        else:
            with closing(self._connect()) as connection:
                if create_first:
                    self._create_table(table, create_sql, connection)

                job = connection.load_table_from_dataframe(df, table)
                job.result()  # Wait for the job to complete.

        logger.info(f"Inserted {len(df)} records into {table}")

    def _create_table(self, table, sql, connection=None):
        """Create table if it doesn't exist.

        Parameters
        ----------
        table: str, optional
            project.dataset.table name to create.
        sql: str or path, optional
            SQL query string or path to file with create statements.
        connection: google.cloud.bigquery.client, optional
            client to the database, if it's already created.
        """
        if connection is None:
            connection = self._connect()

        try:
            connection.get_table(table)  # Make an API request.
        except exceptions.NotFound:
            self.execute(sql, params={'table_id': table}, connection=connection)
            logger.info(f"Created {table}")
