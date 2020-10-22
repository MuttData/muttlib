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
    def __init__(
        self,
        db: str,
        table: str,
        project: str,
        auth: Optional[str],
        auth_file: Optional[Union[str, Path]],
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
        self.client = None
        self.db = db
        self.table = table

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
            dict_creds = json.loads(auth, strict=False)
            return service_account.Credentials.from_service_account_info(dict_creds)

    def _connect(self):
        """Create a BigQuery Client

        Returns
        ----------
        google.cloud.bigquery.client
        """
        if self.credentials is None:
            self.credentials = self._read_cred(self.auth, self.auth_file)
        if self.client is None:
            self.client = bigquery.Client(
                project=self.project, credentials=self.credentials
            )
        return self.client

    def close(self):
        """Close connection"""
        if self.client is not None:
            self.client.close()

    def execute(self, sql, params=None):  # pylint: disable=W0613
        """Execute sql statement.

        sql: str or path
            SQL query string or path to file with Select statements.
        params: dict
            parameters to add into the SQL query to execute.
        client: google.cloud.bigquery.client
            client to the database, if it's already created.
        """
        if self.client is None:
            self.client = self._connect()

        sql = utils.path_or_string(sql)
        if params is not None:
            sql = sql.format(**params)
        logger.info(f"Executing query:\n{sql}")
        return self.client.query(sql)

    def to_frame(self, sql, params=None):
        """Return sql execution as Pandas dataframe.

        sql: str or path
            SQL query string or path to file with Select statements.
        params: dict
            parameters to add into the SQL query to execute.
        client: google.cloud.bigquery.client
            client to the database, if it's already created.
        """
        return self.execute(sql, params).to_dataframe()

    def insert_from_frame(
        self, df, create_first=True, create_sql=None,
    ):
        """Insert from a Pandas dataframe. If it is an existing table,
        the schema of the DataFrame must match the schema of the destination table.
        If the table does not yet exist, the schema is inferred from the DataFrame.

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe with data to be inserted.
        create_first: bool
            Whether to create table before attempting insertion.
        create_sql: str or path
            SQL query string or path to file with create statements.
        """
        table_id = f"{self.project}.{self.db}.{self.table}"
        logger.info(f"Going to insert data into {table_id}")

        if create_first:
            self._create_table(table_id, create_sql)

        job = self._connect().load_table_from_dataframe(df, table_id)
        job.result()  # Wait for the job to complete.

        logger.info(f"Inserted {len(df)} records into {table_id}")

    def _create_table(self, table_id, sql):
        """Create table if it doesn't exist.

        Parameters
        ----------
        table_id: str
            project.dataset.table name to create.
        sql: str or path
            SQL query string or path to file with create statements.
        """
        try:
            self._connect().get_table(table_id)  # Make an API request.
        except exceptions.NotFound:
            self.execute(sql, params={'table_id': table_id})
            logger.info(f"Created {table_id}")
