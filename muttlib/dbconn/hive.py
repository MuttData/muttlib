from contextlib import closing
import logging
import re
from time import sleep

import pandas as pd
import progressbar

import muttlib.utils as utils

logger = logging.getLogger(__name__)
try:
    from TCLIService.ttypes import TOperationState  # noqa: F401 # pylint: disable=W0611
    from pyhive import hive
except ModuleNotFoundError:
    logger.debug("No Hive support.")

HIVE_DB_TYPE = 'hive'


class HiveClient:
    """Wrapper around PyHive's hive module.

    Parameters
    ----------
    host: str
        Host url.
    port: int, Optional
        Port to connect to the HiveServer. Defaults to 10000.
    auth: str, Optional
        Authentication protocol or layer. Defaults to "NOSASL".
    database: str, Optional
        Name of database to connect to.
    username: str, Optional
        Hive username.
    password: str, Optional
        Use with `auth="LDAP"` or `auth="CUSTOM"` only.

    Notes
    ----------
    Refer to PyHive's docstrings for better context on parameters' descriptions:
    https://github.com/dropbox/PyHive/blob/2c2446bf905ea321aac9dcdd3fa033909ff0b0b5/pyhive/hive.py#L105

    """

    def __init__(
        self,
        host,
        port=10_000,
        auth='NOSASL',
        database='default',
        username=None,
        password=None,
    ):
        self.host = host
        self.port = port
        self.auth = auth
        self.database = database
        self.username = username
        self.password = password

    def _connect(self):
        """Instance a connection to the database.

        Returns
        ----------
        pyhive.hive.Connection
        """
        return hive.connect(
            host=self.host,
            port=self.port,
            auth=self.auth,
            database=self.database,
            username=self.username,
            password=self.password,
        )

    def _cursor(self):
        """Instance a cursor from HiveServer connection.

        Returns
        ----------
        pyhive.hive.Cursor
        """
        conn = self._connect()
        return conn.cursor()

    def execute(
        self, sql, params=None, show_progress=True, dry_run=False, async_=False
    ):
        """Execute sql statement.

        Returns
        ----------
        pyhive.hive.Cursor or None
        """
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
        """
        Execute sql statement and return results as a Pandas dataframe.

        Returns
        ----------
        pd.DataFrame
        """
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


# Backward compatiblity alias.
# TODO: Deprecate this.
HiveDb = HiveClient
