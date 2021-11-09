import logging

from muttlib.dbconn.base import EngineBaseClient

logger = logging.getLogger(__name__)

REDSHIFT_DB_TYPE = "redshift"
REDSHIFT_DIALECT = "redshift"

try:
    import psycopg2  # noqa: F401 # pylint:disable=unused-import
except ModuleNotFoundError:
    logger.debug("No Redshift support.")


class RedshiftClient(EngineBaseClient):
    """Create redshift DB client."""

    default_dialect = REDSHIFT_DIALECT
