"""Jinja filters."""

from numbers import Number
from typing import Dict, List, Tuple, Union, Callable

from muttlib.jinja.exceptions import BadInClauseException


def _format_value_in_clause(value: Union[Number, str]) -> str:
    """Format value according to type.

    Args:
        value (Union[Number, str]): any number or string

    Raises:
        BadInClauseException: for values other than Number or str

    Returns:
        str: formatted string
    """
    if isinstance(value, str):
        return f"'{value}'"
    elif isinstance(value, Number):
        return f"{value}"
    else:
        raise BadInClauseException(
            f"Value type: {type(value)} is not allowed for in clause formatting"
        )


def inclause(
    iterable: Union[Tuple[Union[Number, str]], List[Union[Number, str]]]
) -> str:
    """
    Create a Jinja2 filter to format list-like values passed.

    Args:
        iterable (list, tuple): list / tuple of strings and numbers. Can be empty.

    Raises:
        BadInClauseException: for non iterable inputs.

    Returns:
        str: The formatted string of the list of elements.

    Notes:
        Idea originally from
        https://github.com/hashedin/jinjasql/blob/master/jinjasql/core.py
        Passing an empty tuple/list won't raise an exception, in order to
        simplify the function. Also, regarding failing queries, there's no
        explicit goal for sql formatting (although that's a common use case).

    Examples:
        >>> format_in_clause([1.12, 1, 'a'])

        (1.12,1,'a')
    """
    if not isinstance(iterable, (list, tuple)):
        raise BadInClauseException(
            f"Value passed is not a list or tuple: '{iterable}'. "
            f"Where the query uses the '| inclause'."
        )
    values = [_format_value_in_clause(v) for v in iterable]
    clause = ", ".join(values)
    clause = "(" + clause + ")"
    return clause


def quote(iterable: List) -> List[str]:
    """Quote elements of list."""
    return [f"'{x}'" for x in iterable]


__all_filters__: Dict[str, Callable] = {"quote": quote, "inclause": inclause}
