import logging
from pathlib import Path
from typing import Any, Dict, Union, Callable, Optional

import jinja2
from jinjasql import JinjaSql

from muttlib.utils import path_or_string
from muttlib.jinja.filters import __all_filters__

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


DEFAULT_JINJA_ENV_ARGS = {
    "autoescape": True,
    "line_statement_prefix": "%",
    "trim_blocks": True,
    "lstrip_blocks": True,
}


def get_default_jinja_template(
    path_or_str: Union[Path, str],
    filters: Optional[Dict[str, Callable]] = None,
    **kwargs,
):
    """Create Jinja specific template.."""
    if filters is None:
        filters = __all_filters__
    environment = jinja2.Environment(**{**DEFAULT_JINJA_ENV_ARGS, **kwargs})
    environment.filters = {**environment.filters, **filters}
    return environment.from_string(path_or_string(path_or_str))


def load_sql_query(
    sql: Union[Path, str], query_context_params: Dict[str, Any], filters: Dict, **kwargs
) -> str:
    """Render jinja sql with params.

    Parameters
    ----------
    sql : str
        Query to be formatted.
    query_context_params : Dict
        Dict that contains parameters to format the query.

    Returns
    -------
    sql : str
        Formatted query.
    """
    query = path_or_string(sql)

    environment = jinja2.Environment(**{**DEFAULT_JINJA_ENV_ARGS, **kwargs})
    environment.filters = {**environment.filters, **filters}

    if query_context_params:
        j = JinjaSql(env=environment, param_style="pyformat")
        binded_sql, bind_params = j.prepare_query(query, query_context_params)
        missing_placeholders = [
            k for k, v in bind_params.items() if jinja2.Undefined() == v
        ]

        assert (
            len(missing_placeholders) == 0
        ), f"Missing placeholders are: {missing_placeholders}"

        try:
            query = binded_sql % bind_params
        except KeyError as exc:
            logger.error(exc)
            raise

    return query
