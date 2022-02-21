#!/usr/bin/env python
"""Check and fix 'all' extra"""

import sys
import logging
import pathlib

import toml as libtoml

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


TOML = "pyproject.toml"


def main():
    toml_loc = pathlib.Path(__name__).parent / TOML
    with open(toml_loc.resolve()) as f:
        toml_s = f.read()
    toml = libtoml.loads(toml_s)
    try:
        all_extras_deps = [
            dep
            for extra_name, deps in toml["tool"]["poetry"]["extras"].items()
            if extra_name != "all"
            for dep in deps
        ]
        toml["tool"]["poetry"]["extras"]["all"] = sorted(all_extras_deps)
        with open(toml_loc, "w") as f:
            libtoml.dump(toml, f, encoder=libtoml.TomlPreserveInlineDictEncoder())
    except Exception:  # pylint: disable=broad-except
        logger.error("Error fixing pyproject.toml", exc_info=True)
        with open(toml_loc.resolve(), "w") as f:
            f.write(toml_s)
        sys.exit(1)


if __name__ == "__main__":
    main()
