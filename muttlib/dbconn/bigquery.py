import json
import logging
from pathlib import Path
from typing import Any, Dict, Union, Optional

import pandas as pd
from google.cloud import bigquery

import muttlib.utils as utils
from muttlib.dbconn.base import BaseClient
from muttlib.jinja.templating import load_sql_query

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
try:
    import google.cloud.bigquery as bigquery
    from google.oauth2 import service_account
    import google.cloud.exceptions as exceptions
except ModuleNotFoundError:
    logger.debug("No BigQuery support.")


BIGQUERY_DB_TYPE = 'bigquery'


class BigQueryClient(BaseClient):
    """BigQuery Client."""

    def __init__(
        self,
        db: Optional[str] = None,
        project: Optional[str] = None,
        table: Optional[str] = None,
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
    def table_id(self) -> Optional[str]:
        """Get table id."""
        if self.table is not None:
            return f"{self.project}.{self.db}.{self.table}"
        return None

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

    def __del__(self):
        if self.connection is not None:
            self.connection.close()

    def _connect(self) -> bigquery.client.Client:
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

    def execute(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
        connection: Optional[bigquery.client.Client] = None,
        mode='python',
        *,
        dry_run: bool = True,
    ):
        if dry_run:
            self._connect()
            conn: bigquery.client.Client = connection or self._connect()

            job_config = bigquery.QueryJobConfig()
            job_config.dry_run = True

            query_job = conn.query(sql, job_config)

            # This will only work for native tables
            logger.info(
                f"Total data processed: {utils.humansize(query_job.total_bytes_processed)}"
            )

            return sql

        return super().execute(sql, params, connection, mode, dry_run=dry_run)

    def to_frame(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
        connection: Optional[bigquery.client.Client] = None,
        mode='python',
        *,
        dry_run: bool = False,
    ) -> pd.DataFrame:  # pylint: disable=W0221
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
        query_result = self.execute(sql, params, connection, mode, dry_run=dry_run)

        if dry_run:
            return query_result

        df = query_result.to_dataframe()

        logger.info(
            f"Total data processed: {utils.humansize(query_result.total_bytes_processed)}"
        )
        logger.info(
            f"Total data billed: {utils.humansize(query_result.total_bytes_billed)}"
        )
        logger.info(f"{len(df)} rows retrieved")

        return df

    def insert_from_frame(  # pylint: disable=W0221
        self,
        df: pd.DataFrame,
        table: str = None,
        create_first: bool = True,
        create_sql: Optional[Union[str, Path]] = None,
        connection: Optional[bigquery.client.Client] = None,
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
            if self.table_id is None:
                raise ValueError(
                    "This connector has no table set, so you need to pass a table to insert into"
                )
            table = self.table_id

        logger.info(f"Going to insert data into {table}")

        conn = connection or self._connect()
        if create_first:
            if create_sql is None:
                raise ValueError("Need to pass sql to create table")
            self._create_table(table, create_sql, conn)

            job = conn.load_table_from_dataframe(df, table)
            job.result()  # Wait for the job to complete.

        logger.info(f"Inserted {len(df)} records into {table}")

    def _create_table(
        self,
        table: str,
        sql: Union[str, Path],
        connection: Optional[bigquery.client.Client] = None,
    ):
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

        sql_str: str = utils.path_or_string(sql)

        try:
            connection.get_table(table)  # Make an API request.
        except exceptions.NotFound:
            self.execute(sql_str, params={'table_id': table}, connection=connection)
            logger.info(f"Created {table}")
