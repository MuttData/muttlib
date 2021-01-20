"""Athena client."""

import logging
from urllib.parse import quote_plus

from muttlib.dbconn.base import EngineBaseClient
from sqlalchemy.engine.url import URL

logger = logging.getLogger(__name__)

try:
    import pyathena  # noqa: F401 # pylint:disable=unused-import
except ModuleNotFoundError:
    logger.debug("No pyathena support.")

ATHENA_DRIVER = "awsathena+rest"


class AthenaClient(EngineBaseClient):
    """AWS Athena client."""

    def __init__(
        self,
        aws_access_key_id,
        aws_secret_access_key,
        region,
        s3_staging_dir,
        database="default",
        port=443,
    ):
        self.conn_url = URL(
            drivername=ATHENA_DRIVER,
            host=f"athena.{region}.amazonaws.com",
            port=port,
            username=quote_plus(aws_access_key_id),
            password=quote_plus(aws_secret_access_key),
            database=database,
            query={"s3_staging_dir": quote_plus(s3_staging_dir)},
        )
