class JinjaTemplateException(Exception):
    """Jinja template exception."""


class BadInClauseException(JinjaTemplateException):
    """Bad in clause"""
