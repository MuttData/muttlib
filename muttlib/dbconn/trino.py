import logging

from muttlib.dbconn.base import EngineBaseClient

logger = logging.getLogger(__name__)

TRINO_DB_TYPE = "trino"
TRINO_DIALECT = "trino"

try:
    import trino  # noqa: F401 # pylint:disable=unused-import
except ModuleNotFoundError:
    logger.debug("No Trino support.")


class TrinoClient(EngineBaseClient):
    """Create Trino DB client."""

    default_dialect = TRINO_DIALECT
