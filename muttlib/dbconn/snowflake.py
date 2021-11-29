import logging

from snowflake.sqlalchemy import URL
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PublicFormat,
    PrivateFormat,
    BestAvailableEncryption,
    load_pem_private_key,
)

from muttlib.dbconn.base import EngineBaseClient

logger = logging.getLogger(__name__)

SNOWFLAKE_DB_TYPE = "snowflake"


def read_private_key(
    file: bytes,
    passphrase: str,
    encoding_type: str = "DER",
    format_private: bool = True,
    format_type: str = "PKCS8",
    encryption: bool = False,
) -> bytes:
    """Read Private Key

    Parameters
    ----------
    file : bytes
        Private Key file.
    passphrase : str
        Private Key passphrase.
    encoding_type : str, optional
        PK Encoding Type, by default "DER"
    format_private : bool, optional
        PK format private, by default True
    format_type : str, optional
        PK format type, by default "PKCS8"
    encryption : bool, optional
        PK encryption, by default False

    Returns
    -------
    bytes
        Private Key.

    Raises
    ------
    AttributeError
        If Invalid private key encoding type.
    AttributeError
        If Invalid private key format.
    Exception
        If Failed to load private key file
    """
    try:
        pk_encoding_type = Encoding.__getattr__(encoding_type)
    except AttributeError:
        raise AttributeError(f"Invalid private key encoding: '{encoding_type}'")

    try:
        f = PrivateFormat if format_private else PublicFormat
        pk_format_type = f.__getattr__(format_type)
    except AttributeError:
        raise AttributeError(f"Invalid private key format: '{format_type}'")

    pk_encryption = BestAvailableEncryption() if encryption else NoEncryption()

    try:
        pk_file = load_pem_private_key(
            file, password=passphrase.encode(), backend=default_backend(),
        )
        private_key = pk_file.private_bytes(
            encoding=pk_encoding_type,
            format=pk_format_type,
            encryption_algorithm=pk_encryption,
        )
    except Exception:
        logger.error("Failed to load private key file.", exc_info=True)
        raise Exception

    return private_key


class SnowflakeClient(EngineBaseClient):

    default_dialect = "snowflake"

    def __init__(
        self,
        account: str = None,
        warehouse: str = None,
        schema: str = None,
        role: str = "",
        password: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.account = account
        self.warehouse = warehouse
        self.schema = schema
        self.role = role
        self.password = password

    @property
    def conn_str(self):
        return str(
            URL(
                account=self.account,
                user=self.username,
                database=self.database,
                password=self.password,
                schema=self.schema,
                warehouse=self.warehouse,
                role=self.role,
            )
        )
