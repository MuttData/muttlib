"""Module to get and use multiple Big Data DB connections."""
import logging
import os
import re
import shutil
from functools import wraps
from os import makedirs
from time import sleep
from urllib.parse import urlparse

import jinja2
import pandas as pd
from pandas.io.json import json_normalize
import progressbar
import pyarrow.parquet as pq
from sqlalchemy import create_engine
from sqlalchemy.types import VARCHAR


import utils

logger = logging.getLogger(f'dbconn.{__name__}') # NOQA

try:
    import cx_Oracle
except:
    logger.warning("No Oracle support.")

try:
    from TCLIService.ttypes import TOperationState
except:
    logger.warning("No Hive support.")

try:
    import pymongo
except:
    logger.warning("No Mongo support.")


try:
    import ibis
except:
    logger.warning("No Ibis support.")


def _parse_sql_statement_decorator(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        args = list(args)
        sql = utils.path_or_string(args[0])
        format_params = kwargs.get('params', None)
        if format_params:
            try:
                sql = sql.format(**format_params)
            except KeyError as e:
                if e not in format_params:
                    # If the sql string has an unformatted key then fail
                    raise
                else:
                    pass

        logger.debug(f"Running the following query: \n{sql}")
        args[0] = sql
        return func(self, *args, **kwargs)

    return wrapper


class BaseClient:
    """Create BaseClient for DBs."""

    def __init__(
        self,
        username,
        database=None,
        host=None,
        dialect=None,
        port=None,
        driver=None,
        password=None,
    ):
        self.dialect = dialect
        self.host = host
        self.username = username
        self.database = database
        self.port = port
        self.driver = driver
        self.password = password
        self._engine = None

    @property
    def _db_uri(self):
        dialect = (
            f"{self.dialect}{f'+{self.driver}' if self.driver is not None else ''}"
        )
        login = (
            f"{self.username}{f':{self.password}' if self.password is not None else ''}"
        )
        host = f"{self.host}{f':{self.port}' if self.port is not None else ''}"
        return f'{dialect}://{login}@{host}/{self.database}'

    def _get_engine(self, custom_uri=None, connect_args=None, echo=False):
        connect_args = {} if connect_args is None else connect_args
        if not self._engine:
            db_uri = custom_uri or self._db_uri
            self._engine = create_engine(db_uri, connect_args=connect_args, echo=echo)
        return self._engine

    def _connect(self):
        return self._get_engine().connect()

    @staticmethod
    def _cursor_columns(cursor):
        if hasattr(cursor, 'keys'):
            return cursor.keys()
        else:
            return [c[0] for c in cursor.description]

    @_parse_sql_statement_decorator
    def execute(self, sql, params=None, connection=None):
        if connection is None:
            connection = self._connect()
        return connection.execute(sql)

    def to_frame(self, *args, **kwargs):
        cursor = self.execute(*args, **kwargs)
        if not cursor:
            return
        data = cursor.fetchall()
        if data:
            df = pd.DataFrame(data, columns=self._cursor_columns(cursor))
        else:
            df = pd.DataFrame()
        return df

    def insert_from_frame(self, df, table, if_exists='append', index=False, **kwargs):
        # TODO: Validate types here?
        if self.dialect == 'oracle':
            df.columns = [c.upper() for c in df.columns]

        connection = self._connect()
        with connection:
            df.to_sql(table, connection, if_exists=if_exists, index=index, **kwargs)


class PgClient(BaseClient):
    """Create Postgres DB client."""

    def __init__(self, dialect='postgresql', **kwargs):
        super().__init__(dialect=dialect, port=5432, **kwargs)


class OracleClient(BaseClient):
    """Create Oracle DB client."""

    def __init__(self, dialect='oracle', schema=None, **kwargs):
        super().__init__(dialect=dialect, **kwargs)
        self.schema = schema

    def _connect(self):
        dsn = cx_Oracle.makedsn(self.host, self.port, service_name=self.database)
        db_uri = f'{self.dialect}://{self.username}:{self.password}@{dsn}'
        connect_args = {
            'encoding': 'UTF-8',
            'nencoding': 'UTF-8',
        }
        conn = self._get_engine(custom_uri=db_uri, connect_args=connect_args).connect()
        if self.schema is not None:
            conn.connection.current_schema = self.schema
        return conn

    def insert_from_frame(self, df, table, fix_clobs=True, upper_case_cols=True, **kwargs):
        """
        Fix columns case and cast CLOBs to VARCHAR o avoid slow inserts.

        Ref:
        - https://stackoverflow.com/questions/42727990/speed-up-to-sql-when-writing-pandas-dataframe-to-oracle-database-using-sqlalch
        - https://docs.sqlalchemy.org/en/latest/dialects/oracle.html#fine-grained-control-over-cx-oracle-data-binding-and-performance-with-setinputsizes
        """
        # TODO: Validate types here?

        if fix_clobs:
            dtyp = {
                c: VARCHAR(df[c].str.len().max())
                for c in df.columns[df.dtypes == 'object'].tolist()
            }
        else:
            dtyp = df.dtypes

        if upper_case_cols:
            df.columns = [c.upper() for c in df.columns]

        super().insert_from_frame(df, table, **kwargs)



class IbisClient:
    """Create Ibis/Impala DB Client with Hdfs file-read support."""

    def __init__(
        self,
        host,
        port=None,
        username=None,
        hdfs_host=None,
        hdfs_port=None,
        hdfs_username=None,
        hdfs_database=None,
        max_retries=5,
        max_backoff=64,
        timeout=120,
        options={'SYNC_DDL': True},
    ):
        self.host = host
        self.port = port if port is not None else 21050
        self.username = username
        self.hdfs_host = hdfs_host
        self.hdfs_port = hdfs_port
        self.hdfs_username = hdfs_username
        self.hdfs_database = hdfs_database
        self.hdfs_client = None
        self._max_retries = max_retries
        self._max_backoff = max_backoff
        self._timeout = timeout
        self.options = options

    def _connect(self):
        if (
            self.hdfs_host is not None
            and self.hdfs_port is not None
            and self.hdfs_username is not None
        ):
            self.hdfs_client = ibis.hdfs_connect(
                host=self.hdfs_host, port=self.hdfs_port, user=self.hdfs_username
            )

        client = ibis.impala.connect(
            host=self.host,
            port=self.port,
            user=self.username,
            hdfs_client=self.hdfs_client,
            timeout=self._timeout,
        )
        client.set_options(self.options)
        return client

    def execute(self, connection, sql):
        return connection.raw_sql(sql, results=True)

    @_parse_sql_statement_decorator
    def to_frame(
        self,
        sql,
        params=None,
        via_hdfs=True,
        cache_dir=None,
        table_prefix="ibis",
        refresh_cache=False,
        erase_cache_dir=False,
    ):
        # Note: we decorate this func and not the execute to have the formatted sql
        # here and be able to hash it
        connection = self._connect()
        if via_hdfs:
            if self.hdfs_client is None:
                raise ValueError("No hdfs client found!")
            if cache_dir is None:
                raise ValueError("No local dir to save hdfs files was specified!")
            tmp_table = f'{table_prefix}_tmp_{utils.hash_str(sql)}'
            local_tmp_table_dir = cache_dir / tmp_table

            if refresh_cache:
                shutil.rmtree(local_tmp_table_dir, ignore_errors=True)
            if not local_tmp_table_dir.exists():
                self._create_tmp_table(connection, sql, tmp_table)
                self._get_hdfs_data(connection, tmp_table, local_tmp_table_dir)

            df = pq.read_table(local_tmp_table_dir).to_pandas()

            if erase_cache_dir:
                shutil.rmtree(local_tmp_table_dir, ignore_errors=True)
        else:
            cursor = self.execute(sql=sql, connection=connection)
            data = cursor.fetchall()
            if data:
                df = pd.DataFrame(data)
                df.columns = [c[0] for c in cursor.description]
            else:
                df = pd.DataFrame()
            cursor.release()
        return df

    def _create_tmp_table(self, connection, sql, tmp_table):
        logger.debug(f"Creating {tmp_table}...")
        for i in range(1, self._max_retries + 1):
            try:
                create_stmt = f"""
                CREATE TABLE IF NOT EXISTS {self.hdfs_database}.{tmp_table}
                STORED AS parquet AS
                SELECT * FROM (\n{sql}\n LIMIT 1) AS aux_{tmp_table}
                """
                self.execute(connection, create_stmt)
                logger.debug(f"Populating {tmp_table}...")
                insert_stmt = f"""
                INSERT OVERWRITE {self.hdfs_database}.{tmp_table}
                SELECT * FROM (\n{sql}\n) AS aux_{tmp_table}
                """
                self.execute(connection, insert_stmt)
                return
            except Exception as e:
                logger.error(e)
                if i < self._max_retries:
                    backoff_time = min(self._max_backoff, 2 ** i)
                    logger.debug(
                        f"SQL create/insert has failed {i} times. Retrying in "
                        f"{backoff_time} seconds."
                    )
                    sleep(backoff_time)
                else:
                    raise ValueError(f"SQL create/insert failed {i} times. Aborting!")

    def _get_hdfs_data(self, connection, tmp_table, local_tmp_dir):
        """Download hdfs parquet files to local temporary dir.

        Uses incremental backoff sleeping to patiently retry/wait until all files are
        thoroughly downloaded.
        """
        hdfs_files = (
            connection.table(f'{self.hdfs_database}.{tmp_table}')
            .files()['Path']
            .tolist()
        )
        hdfs_dir = urlparse(hdfs_files[0]).path.rsplit('/', 1)[0]
        logger.debug(f"Downloading data from {hdfs_dir}...")
        if not local_tmp_dir.parent.exists():
            makedirs(local_tmp_dir.parent)
        # Dirty incr backoff to fix weird cases in which hdfs doesn't download the
        # parquet files created previously
        try:  # try/finally to iterate download tries and remove tmp table
            for i in range(1, self._max_retries + 1):
                try:  # try/except to download data, catch exception and retry
                    connection.hdfs.get(
                        hdfs_dir, local_path=local_tmp_dir, overwrite=True, verbose=3
                    )
                except Exception as e:
                    logger.error(e)
                if local_tmp_dir.exists():
                    # check if files were downloaded to the local tmp table dir
                    break
                elif i < self._max_retries:
                    backoff_time = min(self._max_backoff, 2 ** i)
                    logger.debug(
                        f"HDFS parquet-get has failed {i} times. Retrying in "
                        f"{backoff_time} seconds."
                    )
                    sleep(backoff_time)
                else:
                    raise ValueError("HDFS parquet-get files failed after {i} tries.")
        finally:
            connection.drop_table(tmp_table, force=True)
            return


class HiveDb:
    """Create Hive DB Client."""

    def __init__(
        self, host, port=None, auth='NOSASL', database='default', username=None
    ):
        self.host = host
        self.port = port if port is not None else 21050
        self.auth = auth
        self.database = database
        self.username = username

    def _connect(self):
        return hive.connect(
            host=self.host,
            port=self.port,
            auth=self.auth,
            database=self.database,
            username=self.username,
        )

    def _cursor(self):
        conn = self._connect()
        return conn.cursor()

    def execute(self, sql, params=None, show_progress=True, dry_run=False):
        sql = utils.path_or_string(sql)
        if params is not None:
            try:
                sql = sql.format(**params)
            except KeyError as e:
                if e not in params:
                    # If the sql string has an unformatted key then fail
                    raise
                else:
                    pass
        if dry_run:
            logger.debug(f"Query dry-run:{sql}")
            return

        cursor = self._cursor()
        cursor.execute(sql, async_=True)

        if show_progress:
            self._show_query_progress(cursor)
        return cursor

    def _show_query_progress(self, cursor, max_val=100, poll_interval=1):
        from TCLIService.ttypes import TOperationState

        # TODO: Add timer logging
        status = cursor.poll()
        bar = progressbar.ProgressBar(max_value=max_val)
        while status.operationState in (
            TOperationState.INITIALIZED_STATE,
            TOperationState.RUNNING_STATE,
        ):
            progress = status.progressUpdateResponse
            if progress is None:
                progress = self._get_progress_from_logs(cursor)
            bar.update(progress * max_val)
            time.sleep(poll_interval)
            status = cursor.poll()
        bar.finish()

    def _get_progress_from_logs(self, cursor, offset=0):
        progress = None
        logs = cursor.fetch_logs()
        if not logs:
            return progress
        log = logs[offset]
        m = re.search(r'\((\d+).*?(\d+)\)', log)
        if m:
            progress, total = m.groups()
            progress = int(progress) / int(total)
        return progress

    def to_frame(self, *args, **kwargs):
        cursor = self.execute(*args, **kwargs)
        if not cursor:
            return
        data = cursor.fetchall()
        # TODO: Add variant that dumps per row rather than the whole thing
        if data:
            df = pd.DataFrame(data)
            df.columns = [c[0] for c in cursor.description]
        else:
            df = pd.DataFrame()
        return df


class SqlServerClient(BaseClient):
    def __init__(self, dialect='mssql', **kwargs):
        super().__init__(dialect=dialect, driver='pymssql', **kwargs)
        if self.database is None:
            raise ValueError("Database argument is not optional!")


class MongoClient:
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

    def _connect(self, custom_uri=None):
        if custom_uri is None:
            conn_uri = self._build_conn_uri()
        client = pymongo.MongoClient(conn_uri)
        return client[self.database]

    def _find(self, collection, query=None, fields=None, db=None, limit=0):
        if db is None:
            db = self._connect()
        collection = db[collection]
        find_func = getattr(collection, 'find')
        res = find_func(query, fields)
        if limit > 0:
            res = res.limit(limit)
        return list(res)

    def to_frame(self, collection, query=None, fields=None, limit=0, no_id=False):
        cursor = self._find(collection, query=query, fields=fields, limit=limit)
        df = json_normalize(cursor)
        if no_id and not df.empty:
            del df['_id']
        return df

    def insert(self, collection, query=None, db=None):
        if db is None:
            db = self._connect()
        collection = db[collection]
        insert_id = collection.insert_one(query).inserted_id
        return insert_id


ORACLE_DB_TYPE = 'oracle'
POSTGRES_DB_TYPE = 'postgres'
SQLSERVER_DB_TYPE = 'sql_server'
HIVE_DB_TYPE = 'hive'
MONGO_DB_TYPE = 'mongo'

connectors = {
    ORACLE_DB_TYPE: OracleClient,
    POSTGRES_DB_TYPE: PgClient,
    SQLSERVER_DB_TYPE: SqlServerClient,
    HIVE_DB_TYPE: HiveDb,
    MONGO_DB_TYPE: MongoClient,
}


def get_client(creds):
    db_type = creds.pop('db_type')
    return db_type, connectors[db_type](**creds)
