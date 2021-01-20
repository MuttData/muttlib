"""Presto Client."""

from contextlib import closing
import logging
import re
from time import sleep

import pandas as pd
import progressbar

from muttlib.dbconn.base import BaseClient
import muttlib.utils as utils

logger = logging.getLogger(__name__)
try:
    from TCLIService.ttypes import TOperationState  # noqa: F401 # pylint: disable=W0611
    from pyhive import presto as pyhive_presto
except ModuleNotFoundError:
    logger.debug("No Presto support.")


class PrestoClient(BaseClient):
    """Wrapper around PyHive's Presto module.

    Parameters
    ----------
    host: str
        Host url.
    port: int, Optional
        Port to connect to the Presto server. Defaults to 443.
    """

    def __init__(
        self,
        host,
        port=443,
        protocol="https",
        catalog="hive",
        schema="default",
        username=None,
        password=None,
        requests_kwargs=None,
    ):
        super().__init__(
            host=host, port=port, username=username, password=password,
        )
        self.protocol = protocol
        self.catalog = catalog
        self.schema = schema
        self.requests_kwargs = requests_kwargs

    def _connect(self):
        """Instance a connection to the database."""
        return pyhive_presto.connect(
            host=self.host,
            port=self.port,
            username=self.username,
            catalog=self.catalog,
            schema=self.schema,
            protocol=self.protocol,
            requests_kwargs=self.requests_kwargs,
        )

    def execute(
        self, sql, params=None, connection=None, show_progress=True, dry_run=False,
    ):
        """Execute sql statement."""
        sql = utils.path_or_string(sql)
        if params is not None:
            sql = sql.format(**params)

        if dry_run:
            logger.debug(f"Query dry-run:{sql}")
            return

        should_close = False

        if connection is None:
            connection = self._connect()
            should_close = True

        cursor = connection.cursor()
        cursor.execute(sql)

        if show_progress:
            self._show_query_progress(cursor)

        if should_close:
            connection.close()

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


# Backward compatiblity alias.
# TODO: Deprecate this.
PrestoDb = PrestoClient
