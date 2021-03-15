import logging
import shutil
from time import sleep
from urllib.parse import urlparse

import pandas as pd

from muttlib.dbconn.base import parse_sql_statement_decorator
import muttlib.utils as utils

logger = logging.getLogger(__name__)

try:
    import ibis
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    logger.debug("No Ibis support.")


IBIS_DB_TYPE = 'ibis'


class IbisClient:
    """Create Ibis/Impala DB Client with Hdfs file-read support.

    Parameters
    ----------
    host : str
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
        temp_db=None,
        temp_hdfs_path=None,
        timeout=None,
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
            with ibis.config.config_prefix('impala'):
                if temp_db is not None:
                    ibis.config.set_option('temp_db', temp_db)
                if temp_hdfs_path is not None:
                    ibis.config.set_option('temp_hdfs_path', temp_hdfs_path)

    def _connect(self):
        if (
            self.hdfs_host is not None
            and self.hdfs_port is not None
            and self.hdfs_username is not None
        ):
            self.hdfs_client = ibis.impala.hdfs_connect(
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

    def execute(self, sql, client=None, return_cursor=False):
        """Execute raw sql statement."""
        if client is None:
            client = self._connect()
        return client.raw_sql(sql, results=return_cursor)

    @parse_sql_statement_decorator
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
                    self.execute(drop_stmt, client)
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
                self.execute(drop_stmt, client)

                logger.debug(f"Creating {tmp_table}...")
                create_sql = """
                CREATE TABLE IF NOT EXISTS {{hdfs_database}}.{{tmp_table}}
                STORED AS parquet AS
                SELECT * FROM (\n{{sql}}\n LIMIT 1) AS aux_{{tmp_table}}
                """
                create_stmt = utils.get_default_jinja_template(create_sql).render(
                    **{
                        'hdfs_database': self.hdfs_database,
                        'tmp_table': tmp_table,
                        'sql': sql,
                    }
                )

                self.execute(create_stmt, client)

                logger.debug(f"Populating {tmp_table}...")
                insert_sql = """
                INSERT OVERWRITE {{hdfs_database}}.{{tmp_table}}
                SELECT * FROM (\n{{sql}}\n) AS aux_{{tmp_table}}
                """
                insert_stmt = utils.get_default_jinja_template(insert_sql).render(
                    **{
                        'hdfs_database': self.hdfs_database,
                        'tmp_table': tmp_table,
                        'sql': sql,
                    }
                )
                self.execute(insert_stmt, client)
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
