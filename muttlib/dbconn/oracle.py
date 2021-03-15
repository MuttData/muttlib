import logging

from sqlalchemy.types import VARCHAR

from muttlib.dbconn.base import EngineBaseClient

logger = logging.getLogger(__name__)


try:
    import cx_Oracle
except ModuleNotFoundError:
    logger.debug("No Oracle support.")

ORACLE_DB_TYPE = 'oracle'


class OracleClient(EngineBaseClient):
    """Create Oracle DB client."""

    default_dialect = 'oracle'

    def __init__(self, schema=None, **kwargs):
        super().__init__(**kwargs)
        self.schema = schema

    @property
    def conn_str(self):
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
