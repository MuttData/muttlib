import logging

from muttlib.dbconn.base import BaseClient

logger = logging.getLogger(__name__)

MYSQL_DB_TYPE = 'mysql'


try:
    import pymysql  # noqa: F401 # pylint:disable=unused-import
except ModuleNotFoundError:
    logger.debug("No MySql support.")


class MySqlClient(BaseClient):
    """Create MySql DB client."""

    default_driver = 'pymysql'
    default_dialect = 'mysql'
