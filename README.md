# muttlib 🐶📚

[![pypi version](https://img.shields.io/pypi/v/muttlib?color=blue)](https://pypi.org/project/muttlib/)
[![pipeline status](https://gitlab.com/mutt_data/muttlib/badges/master/pipeline.svg)](https://gitlab.com/mutt_data/muttlib/-/commits/master)
[![coverage report](https://gitlab.com/mutt_data/muttlib/badges/master/coverage.svg)](https://gitlab.com/mutt_data/muttlib/-/commits/master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
<!-- TODO: fix for github -->
<!-- [![docstring report](https://gitlab.com/mutt_data/muttlib/-/jobs/artifacts/master/raw/docs_coverage.svg?job=docstr-cov)](https://interrogate.readthedocs.io/en/latest/) -->

## Description

Library with helper code to start a data-related project.
By [Mutt Data](https://muttdata.ai/).

Current modules:

- `dbconn`: Somewhat homogeneus lib to access multiple DBs.
- `file_processing`: Helpers for concurrent file processing.
- `forecast`: Provides FBProphet a common interface to Sklearn and general
  utilities for forecasting problems, allowing wider and easier grid search for
  hyperparameters.
- `utils`: A single version of miscellaneous functions needed every now and then.
- `gsheetsconn`: Module to make data interactions to/from Google Sheets <> Pandas.
- `gdrive`: Module that provides a UNIX-ish interface to GDrive.

## Table of Contents

- [muttlib 🐶📚](#muttlib-)
  - [Description](#description)
  - [Table of Contents](#table-of-contents)
    - [Installing from PyPi](#installing-from-pypi)
    - [Installing custom branches from the repos](#installing-custom-branches-from-the-repos)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)

### Installing from PyPI

_Note:_ `muttlib` is packaged and developed using [poetry](https://python-poetry.org).

Base lib:

```bash
poetry add muttlib
```

Check below for available extras.

Parquet and Feather support:

```bash
poetry add muttlib -E pyarrow
```

Forecast:

```bash
poetry add muttlib -E forecast
```

Misc DB support for dbconn:

```bash
poetry add muttlib -E oracle
poetry add muttlib -E hive
poetry add muttlib -E postgres
poetry add muttlib -E mysql
poetry add muttlib -E sqlserver
poetry add muttlib -E mongo
poetry add muttlib -E ibis
```

_Note:_ the `ibis` extra requires installing binary packages. Check [CONTRIBUTING.md](https://github.com/MuttData/muttlib/blob/master/CONTRIBUTING.md#prerequisites) for the full list.

### Installing custom branches from the repos

From GitHub mirror:

```bash
poetry add -e git+https://github.com/MuttData/muttlib.git@AWESOME_FEATURE_BRANCH#egg=muttlib
```

From Gitlab main repo:

```bash
poetry add -e git+https://gitlab.com/mutt_data/muttlib.git@AWESOME_FEATURE_BRANCH#egg=muttlib
```

## Usage

See the [documentation](https://mutt_data.gitlab.io/muttlib/) to get started with `muttlib`.

## Contributing

We appreciate for considering to help out maintaining this project. If you'd like to contribute please read our [contributing guidelines](https://mutt_data.gitlab.io/muttlib/contributing.html).

## License

`muttlib` is licensed under the [Apache License 2.0](https://github.com/MuttData/muttlib/blob/master/LICENCE).
