"""Module to get and use multiple Big Data DB connections."""
import abc
from contextlib import closing
from functools import wraps
import logging
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from contextlib import closing

import muttlib.utils as utils

logger = logging.getLogger(__name__)


def parse_sql_statement_decorator(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        args = list(args)  # type: ignore
        sql = utils.path_or_string(args[0])
        format_params = kwargs.get('params', None)
        if format_params:
            try:
                sql = sql.format(**format_params)
            except KeyError as e:
                if e not in format_params:
                    # If the sql string has an unformatted key then fail
                    raise
                else:
                    pass

        logger.debug(f"Running the following query: \n{sql}")
        args[0] = sql  # type: ignore
        return func(self, *args, **kwargs)

    return wrapper


def format_drivername(dialect: str, driver: Optional[str] = None):
    """Helper function to format the schema part of connection strings.

    Args:
        dialect (str): Database dialect.
        driver (str, optional): Database driver to be used.

    Returns:
        str: Formatted schema generated.
    """
    parts = [dialect]
    if driver is not None and driver != '':
        parts += [driver]
    return '+'.join(parts)


class BaseClient(abc.ABC):
    """Create BaseClient for DBs."""

    default_dialect: Optional[str] = None  # To be defined by subclasses.
    default_driver: Optional[str] = None

    def __init__(
        self,
        username=None,
        database=None,
        host=None,
        dialect=None,
        port=None,
        driver=None,
        password=None,
    ):
        if dialect is None:
            dialect = self.default_dialect

        if driver is None and self.default_driver is not None:
            driver = self.default_driver

        self.conn_url = URL(
            drivername=format_drivername(dialect, driver),
            username=username,
            password=password,
            host=host,
            port=port,
            database=database,
        )

    # This variable is used to list class attributes to be forwarded to
    # `self.conn_url` for backward compatibility.
    # This should be removed in the future as well as the extra setter/getter
    # logic.
    _conn_url_delegated = (
        "username",
        "database",
        "host",
        "dialect",
        "port",
        "driver",
        "password",
    )

    def __getattr__(self, attr):
        if attr in self._conn_url_delegated:
            if attr == 'dialect':
                return self.conn_url.get_backend_name()
            elif attr == 'driver':
                return self.conn_url.get_driver_name()
            else:
                _cls = URL
                target = self.conn_url
        else:
            _cls = object
            target = self

        return _cls.__getattribute__(target, attr)

    def __setattr__(self, attr, value):
        if attr in self._conn_url_delegated:
            _cls = URL
            target = self.conn_url
            if attr in ('dialect', 'driver'):
                if attr == 'dialect':
                    value = format_drivername(value, self.driver)
                elif attr == 'driver':
                    value = format_drivername(self.dialect, value)
                attr = 'drivername'
        else:
            _cls = object
            target = self

        return _cls.__setattr__(target, attr, value)

    @property
    def conn_str(self):
        return str(self.conn_url)

    @abc.abstractmethod
    def _connect(self):
        raise NotImplementedError

    @staticmethod
    def _cursor_columns(cursor):
        # TODO: This can be moved out of the class
        if hasattr(cursor, 'keys'):
            return cursor.keys()
        else:
            return [c[0] for c in cursor.description]

    @parse_sql_statement_decorator
    def execute(self, sql, params=None, connection=None):  # pylint: disable=W0613
        """Execute sql statement."""
        if params is not None:
            sql = sql.format(**params)

        if connection is not None:
            # It's not our job to close the passed connection
            return connection.execute(sql)

        with self._connect() as connection:
            # Create and destroy a connection, as to avoid dangling connections
            return connection.execute(sql)

    def to_frame(self, sql, params=None, connection=None):
        """Execute SQL statement and return as Pandas dataframe."""
        with closing(self.execute(sql, params, connection)) as cursor:
            if not cursor:
                return
            data = cursor.fetchall()
            if data:
                df = pd.DataFrame(data, columns=self._cursor_columns(cursor))
            else:
                df = pd.DataFrame()
            return df

    def insert_from_frame(
        self, df, table, if_exists='append', index=False, connection=None, **kwargs
    ):
        """Insert from a Pandas dataframe."""
        # TODO: Validate types here?

        column_names = df.columns.tolist()
        chunksize = kwargs.get("chunksize", 10_000)

        if if_exists == "fail" or if_exists == "replace":
            raise NotImplementedError

        if index:
            raise NotImplementedError

        insert_stmt = """
            INSERT INTO {table} ({columns})
            VALUES {values_chunk}
        """

        def _to_sql_tuple(d):
            return f"({', '.join(map(str, d.values()))})"

        def df_chunksize_iterator(df, chunksize=10_000):
            for start in range(0, len(df), chunksize):
                yield df[start : start + chunksize]

        for chunk in df_chunksize_iterator(df, chunksize):
            values = map(_to_sql_tuple, chunk.to_dict(orient="records"))
            chunk_insert_stmt = insert_stmt.format(
                table=table, columns=column_names, values_chunk=",\n".join(values)
            )
            self.execute(chunk_insert_stmt, connection=connection)


class EngineBaseClient(BaseClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine = None

    def _connect(self):
        return self.get_engine().connect()

    def get_engine(self, custom_uri=None, connect_args=None, echo=False):
        """Create engine or return existing one."""
        if not self._engine:
            if connect_args is None:
                connect_args = {}
            db_uri = custom_uri or self.conn_str
            self._engine = create_engine(db_uri, connect_args=connect_args, echo=echo)
        return self._engine

    def insert_from_frame(
        self, df, table, if_exists='append', index=False, connection=None, **kwargs
    ):
        """Insert from a Pandas dataframe."""
        # TODO: Validate types here?

        if connection:
            # It's not our job to close the passed connection
            df.to_sql(table, connection, if_exists=if_exists, index=index, **kwargs)
        else:
            with self._connect() as connection:
                # Create and destroy a connection, as to avoid dangling connections
                df.to_sql(table, connection, if_exists=if_exists, index=index, **kwargs)


class ClientBaseClient(BaseClient):
    pass
