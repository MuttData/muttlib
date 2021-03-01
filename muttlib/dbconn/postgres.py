import logging

from muttlib.dbconn.base import EngineBaseClient

logger = logging.getLogger(__name__)

POSTGRES_DB_TYPE = 'postgres'
POSTGRES_DIALECT = "postgresql"

try:
    import psycopg2  # noqa: F401 # pylint:disable=unused-import
except ModuleNotFoundError:
    logger.debug("No Postgresql support.")


class PgClient(EngineBaseClient):
    """Create Postgres DB client."""

    default_dialect = POSTGRES_DIALECT
