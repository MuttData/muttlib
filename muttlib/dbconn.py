"""Module to get and use multiple Big Data DB connections."""
from functools import wraps
import logging
from contextlib import closing
import re
import shutil
from time import sleep
from urllib.parse import urlparse

import pandas as pd
from pandas.io.json import json_normalize
import progressbar
from sqlalchemy import create_engine
from sqlalchemy.types import VARCHAR

import muttlib.utils as utils

logger = logging.getLogger(f'dbconn.{__name__}')  # NOQA

try:
    import cx_Oracle
except ModuleNotFoundError:
    logger.debug("No Oracle support.")

try:
    from TCLIService.ttypes import TOperationState  # noqa: F401 # pylint: disable=W0611
    from pyhive import hive
except ModuleNotFoundError:
    logger.debug("No Hive support.")

try:
    import pymongo
except ModuleNotFoundError:
    logger.debug("No Mongo support.")

try:
    import ibis
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    logger.debug("No Ibis support.")

try:
    import pymysql  # noqa: F401 # pylint:disable=unused-import
except ModuleNotFoundError:
    logger.debug("No MySql support.")

try:
    import psycopg2  # noqa: F401 # pylint:disable=unused-import
except ModuleNotFoundError:
    logger.debug("No Postgresql support.")


def _parse_sql_statement_decorator(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        args = list(args)  # type: ignore
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
        args[0] = sql  # type: ignore
        return func(self, *args, **kwargs)

    return wrapper


class BaseClient:
    """Create BaseClient for DBs."""

    def __init__(
        self,
        username=None,
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

    def get_engine(self, custom_uri=None, connect_args=None, echo=False):
        """Create engine or return existing one."""
        connect_args = {} if connect_args is None else connect_args
        if not self._engine:
            db_uri = custom_uri or self._db_uri
            self._engine = create_engine(db_uri, connect_args=connect_args, echo=echo)
        return self._engine

    def _connect(self):
        return self.get_engine().connect()

    @staticmethod
    def _cursor_columns(cursor):
        if hasattr(cursor, 'keys'):
            return cursor.keys()
        else:
            return [c[0] for c in cursor.description]

    @_parse_sql_statement_decorator
    def execute(self, sql, params=None, connection=None):  # pylint: disable=W0613
        """Execute sql statement."""
        if connection is None:
            connection = self._connect()
        return connection.execute(sql)

    def to_frame(self, *args, **kwargs):
        """Return sql execution as Pandas dataframe."""
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
        """Insert from a Pandas dataframe."""
        # TODO: Validate types here?
        connection = self._connect()
        with connection:
            df.to_sql(table, connection, if_exists=if_exists, index=index, **kwargs)


class PgClient(BaseClient):
    """Create Postgres DB client."""

    def __init__(self, dialect='postgresql', **kwargs):
        super().__init__(dialect=dialect, port=5432, **kwargs)


class MySqlClient(BaseClient):
    """Create MySql DB client."""

    def __init__(self, dialect='mysql', driver='pymysql', **kwargs):
        super().__init__(dialect=dialect, driver=driver, **kwargs)


class SQLiteClient(BaseClient):
    """Create SQLite DB client."""

    def __init__(self, dialect='sqlite', **kwargs):
        super().__init__(dialect=dialect, **kwargs)

    def _connect(self):
        db_uri = f'{self.dialect}:///{self.database}'
        return self.get_engine(custom_uri=db_uri).connect()


class OracleClient(BaseClient):
    """Create Oracle DB client."""

    def __init__(self, dialect='oracle', schema=None, **kwargs):
        super().__init__(dialect=dialect, **kwargs)
        self.schema = schema

    @property
    def _db_uri(self):
        dsn = cx_Oracle.makedsn(  # pylint: disable=I1101
            self.host, self.port, service_name=self.database
        )
        db_uri = f'{self.dialect}://{self.username}:{self.password}@{dsn}'
        return db_uri

    def _connect(self):
        connect_args = {'encoding': 'UTF-8', 'nencoding': 'UTF-8'}
        conn = self.get_engine(connect_args=connect_args).connect()
        if self.schema is not None:
            conn.connection.current_schema = self.schema
        return conn

    def insert_from_frame(
        self,
        df,
        table,
        fix_clobs=True,
        upper_case_cols=True,
        **kwargs,  # pylint: disable=W0221
    ):
        """
        Fix columns case and cast CLOBs to VARCHAR o avoid slow inserts.

        Ref:
        - https://stackoverflow.com/questions/42727990/speed-up-to-sql-when-writing-pandas-dataframe-to-oracle-database-using-sqlalch # noqa
        - https://docs.sqlalchemy.org/en/latest/dialects/oracle.html#fine-grained-control-over-cx-oracle-data-binding-and-performance-with-setinputsizes # noqa
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

        super().insert_from_frame(df, table, dtype=dtyp, **kwargs)


class IbisClient:
    """Create Ibis/Impala DB Client with Hdfs file-read support.

    Parameters
    ----------
    host : str, optional
        Host name of the impalad node or load blancer.
    port : int, optional
        Impala's HiveServer2 port
    username : str, optional
        LDAP user to authenticate
    database : str, optional
        Default database
    hdfs_host : str, optional
        Default host when using hdfs_client
    hdfs_port : int, optional
        Default port when using hdfs_client
    hdfs_username : str, optional
        Default username when using hdfs_client
    hdfs_database : str, optional
        Default database when using hdfs_client
    max_retriers : int, optional
        Number of retires for incremental backoff when querying via_hdfs.
    max_backoff : int, optional
        Max time in seconds to wait before retrying the incr. backoff.s
    timeout : int, optional
        Connection timeout in seconds when communicating with HiveServer2. Defaults to
        `None` where connections will not drop.
    options : dict, optional
        Impala session options such as SYNC_DDL or others.

    Notes
    --------
    It is strongly advised to set the Impala Session option 'SYNC_DDL' to `True` when
    working with an Impala load-balancer. This will make the balancer sync the metadata
    to all other nodes after a DDL statement.
    When using the via_hdfs=True argument in the to_frame method, you should use a
    particular's node hostname or a IP since this argument generates the creation and
    deletion of a temp table and the needed DDL operations are much faster in this way.
    If you need to point to a load balancer, remember to set SYNC_DDL as True.

    See also
    --------
    Problems when using a load-balancer:
    http://mail-archives.apache.org/mod_mbox/impala-issues/201812.mbox/%3CJIRA.13203856.1544572754000.145264.1544572802089@Atlassian.JIRA%3E # noqa

    """

    def __init__(
        self,
        host,
        port=21050,
        username=None,
        database=None,
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
        self.port = port
        self.username = username
        self.database = database
        self.hdfs_host = hdfs_host
        self.hdfs_port = hdfs_port
        self.hdfs_username = hdfs_username
        self.hdfs_database = hdfs_database
        self.hdfs_client = None
        self._max_retries = max_retries
        self._max_backoff = max_backoff
        self._timeout = timeout
        self.options = options

        # Override ibis default tmp db and location (when using a non-root/admin
        # user that cannot write to `/tmp/ibis`)
        # See https://stackoverflow.com/a/47543691/2149400
        if temp_db is not None or temp_hdfs_path is not None:
            with cf.config_prefix('impala'):
                if temp_db is not None:
                    cf.set_option('temp_db', temp_db)
                if temp_hdfs_path is not None:
                    cf.set_option('temp_hdfs_path', temp_hdfs_path)

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
            database=self.database,
            user=self.username,
            hdfs_client=self.hdfs_client,
            timeout=self._timeout,
        )
        client.set_options(self.options)
        return client

    def execute(self, client, sql, return_cursor=False):
        """Execute raw sql statement."""
        return client.raw_sql(sql, results=return_cursor)

    @_parse_sql_statement_decorator
    def to_frame(
        self,
        sql,
        params=None,  # pylint:disable=W0613
        via_hdfs=False,
        cache_dir=None,
        table_prefix="ibis",
        refresh_cache=False,
        erase_cache_dir=False,
    ):
        """Execute sql statement and return results as a Pandas dataframe."""
        # Note: we decorate this func and not the execute to have the formatted sql
        # here and be able to hash it
        client = self._connect()
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
                drop_stmt = f"""
                DROP TABLE IF EXISTS {self.hdfs_database}.{tmp_table}
                """
                try:
                    self._create_tmp_table(client, sql, tmp_table, drop_stmt)
                    self._get_files_from_parquet_table(
                        client, tmp_table, local_tmp_table_dir
                    )
                except Exception as e:
                    logger.error(e)
                    logger.debug(f"Cleanup of local tmp dir '{local_tmp_table_dir}'.")
                    shutil.rmtree(local_tmp_table_dir, ignore_errors=True)
                    raise ValueError("Failed to query or pull parquet files from HDFS.")
                finally:
                    logger.debug(f"Cleanup tmp table '{tmp_table}'.")
                    self.execute(client, drop_stmt)
                    sleep(1)
                    client.drop_table(tmp_table, force=True)

            df = pq.read_table(local_tmp_table_dir).to_pandas()

            if erase_cache_dir:
                shutil.rmtree(local_tmp_table_dir, ignore_errors=True)
        else:
            try:
                cursor = self.execute(sql=sql, client=client, return_cursor=True)
                data = cursor.fetchall()
                if data:
                    df = pd.DataFrame(data)
                    df.columns = [c[0] for c in cursor.description]
                else:
                    df = pd.DataFrame()
                cursor.release()
            finally:
                client.close()
        return df

    def _create_tmp_table(self, client, sql, tmp_table, drop_stmt):
        for i in range(1, self._max_retries + 1):
            try:
                logger.debug(f"Try table drop if exists {tmp_table}...")
                self.execute(client, drop_stmt)

                logger.debug(f"Creating {tmp_table}...")
                create_stmt = f"""
                CREATE TABLE IF NOT EXISTS {self.hdfs_database}.{tmp_table}
                STORED AS parquet AS
                SELECT * FROM (\n{sql}\n LIMIT 1) AS aux_{tmp_table}
                """
                self.execute(client, create_stmt)

                logger.debug(f"Populating {tmp_table}...")
                insert_stmt = f"""
                INSERT OVERWRITE {self.hdfs_database}.{tmp_table}
                SELECT * FROM (\n{sql}\n) AS aux_{tmp_table}
                """
                self.execute(client, insert_stmt)
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

    def _get_files_from_parquet_table(self, client, tmp_table, local_tmp_dir):
        """Download hdfs parquet files to local temporary dir.

        Uses incremental backoff sleeping to patiently retry/wait until all files are
        thoroughly downloaded.
        """
        hdfs_files = (
            client.table(f'{self.hdfs_database}.{tmp_table}').files()['Path'].tolist()
        )
        hdfs_dir = urlparse(hdfs_files[0]).path.rsplit('/', 1)[0]
        logger.debug(f"Downloading data from {hdfs_dir}...")
        if not local_tmp_dir.exists():
            utils.make_dirs(local_tmp_dir)
        # Dirty incr backoff to fix weird cases in which hdfs doesn't download the
        # parquet files created previously
        for i in range(1, self._max_retries + 1):
            try:  # try/except to download data, catch exception and retry
                client.hdfs.get(
                    hdfs_dir, local_path=local_tmp_dir, overwrite=True, verbose=3
                )
            except Exception as e:
                logger.error(e)

            all_globs = local_tmp_dir.glob('*')  # We only check depth = 0
            parquet_files = [f for f in all_globs if f.is_file()]
            if len(parquet_files) > 0:
                # Check if parquets were downloaded to the local tmp table dir
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


class HiveDb:
    """Create Hive DB Client."""

    def __init__(
        self, host, port=10_000, auth='NOSASL', database='default', username=None
    ):
        self.host = host
        self.port = port
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

    def execute(
        self, sql, params=None, show_progress=True, dry_run=False, async_=False
    ):
        """Execute sql statement."""
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
        cursor.execute(sql, async_=async_)

        if show_progress:
            self._show_query_progress(cursor)
        return cursor

    def _show_query_progress(self, cursor, max_val=100, poll_interval=1):
        from TCLIService.ttypes import TOperationState  # pylint: disable=W0621 # noqa

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
            sleep(poll_interval)
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
        """Execute sql statement and return results as a Pandas dataframe."""
        with closing(self.execute(*args, **kwargs)) as cursor:
            if not cursor:
                return
            data = cursor.fetchall()  # pylint: disable=no-member
            # TODO: Add variant that dumps per row rather than the whole thing
            if data:
                df = pd.DataFrame(data)
                df.columns = [
                    c[0] for c in cursor.description  # pylint: disable=no-member
                ]
            else:
                df = pd.DataFrame()
            return df


class SqlServerClient(BaseClient):
    """SQLServer client."""

    def __init__(self, dialect='mssql', **kwargs):
        super().__init__(dialect=dialect, driver='pymssql', **kwargs)
        if self.database is None:
            raise ValueError("Database argument is not optional!")


class MongoClient:
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


ORACLE_DB_TYPE = 'oracle'
POSTGRES_DB_TYPE = 'postgres'
MYSQL_DB_TYPE = 'mysql'
SQLSERVER_DB_TYPE = 'sql_server'
HIVE_DB_TYPE = 'hive'
MONGO_DB_TYPE = 'mongo'

connectors = {
    ORACLE_DB_TYPE: OracleClient,
    POSTGRES_DB_TYPE: PgClient,
    MYSQL_DB_TYPE: MySqlClient,
    SQLSERVER_DB_TYPE: SqlServerClient,
    HIVE_DB_TYPE: HiveDb,
    MONGO_DB_TYPE: MongoClient,
}


def get_client(creds):
    """Get a db client from a credentials dict."""
    db_type = creds.pop('db_type')
    return db_type, connectors[db_type](**creds)
