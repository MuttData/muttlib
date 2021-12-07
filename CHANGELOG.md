# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.10.post1] - 2021-12-06

### Fixed
   - Pin trino version to 0.4.1

## [1.4.10] - 2021-11-29

### Changed
  - Changed nox session of precommit-hooks to be more descriptive

## [1.4.9.post1] - 2021-11-29

### Added
  - Added isort 5 to pre-commit hooks

### Fixed
  - Pin gspread version to 4.0.1 due to API changes in a new version of gspread (gspread_pandas dependency)

## [1.4.8] - 2021-11-25

### Changed

  - Pinned bandit version.

## [1.4.7] - 2021-11-19

### Added
  - Added docstring coverage badge to `README.md`

## [1.4.6] - 2021-11-16

### Changed
  - `numpy` and `pyarrow` version

## [1.4.5] - 2021-11-15

### Fix
  - Fix CI/CD issues.

## [1.4.4] - 2021-11-13

### Fix
  - Typos on README

## [1.4.3] - 2021-11-12

### Added
  -  Added Black badge to `README.md`

## [1.4.2] - 2021-11-11

### Fix
  -  Fix CI linter issues.

## [1.4.1] - 2021-11-11

### Fix
  -  Fix CI issues.

## [1.4.0] - 2021-10-22

### Added
  -  dbconn Redshift client

## [1.3.0] - 2021-10-19

### Added
  -  dbconn Trino client

## [1.2.1] - 2021-08-20

### Fix
  -  `dbconn.snowflake` client Handle empty role

## [1.2.0] - 2021-08-19

### Added
  -  dbconn Snowflake client

## [1.1.6] - 2021-08-18

### Fixed
  - Fix HDFS connection in ibis module/library.
  - Set version 1.4.0 to `ibis-framework[impala]`

### Deleted
  - Delete `ibis-framework` and `impyla` libs.

## [1.1.5] - 2021-07-11

### Fixed
  -  Let GsheetClient pass credentials as str.

## [1.1.4] - 2021-06-11

### Fixed

  - [[Issue #150](https://gitlab.com/mutt_data/muttlib/-/issues/150)] Fix missing doc section for `plotting` module

## [1.1.3] - 2021-05-31

### Fixed

  - Pinned `pystan` version for `fbprophet` to build correctly

## [1.1.2] - 2021-03-19

### Fixed

  - Fixed `TeradataClient` docstring to pass CI pipeline.

## [1.1.1] - 2021-03-18

### Added

  - `interrogate` to check docstring coverage.
  - Use of `interrogate` in CONTRIBUTING.md#docstrings section.

### Changed

  - Pinned SQLAlchemy version.

## [1.1.0] - 2021-03-16

### Added

  - TeradataClient

## [1.0.8] - 2021-03-11

### Added

  - Tests to `EngineBaseClient`.
  - Tests to `SqliteClient`.

## [1.0.7] - 2021-03-05

Update licence reference from MIT to Apache Software.

## [1.0.6] - 2021-03-01

Add tests to `BigQueryClient`

## [1.0.5] - 2021-03-01

Add `BigQuery` client to `dbconn`

## [1.0.4] - 2021-03-01

Unpin `pandas` version from `setup`

## [1.0.3] - 2021-02-26

Fix `dbconn` submodules docstring rendering

## [1.0.2] - 2021-02-23

Added tests to OracleClient

## [1.0.1] - 2021-02-22

Added tests to MongoClient

## [1.0.0] - 2021-02-19

Enforced deprecations.

## [0.35.4] - 2021-02-12

Added tests for `IbisClient`. Deprecated `IbisClient.execute` in favor of `IbisClient.execute_new`, which will conform to the rest of the client.

## [0.35.3] - 2021-02-11

Add docstrings to `file_processing.py`.

## [0.35.2] - 2021-02-09

Added tests for HiveClient.

## [0.35.1] - 2021-02-08

Added release job to `GitLab CI`.

## [0.34.1] - 2021-02-05

Added pypi version badge to `README.md`

## [0.34.0] - 2021-02-04

Added `get_default_jinja_template` function, which is meant to replace `template`.

## [0.33.2] - 2021-02-02

Fix module dosctring formatting in `forecast.py`.

Pin Numpy version to fix a bug with PyArrow (https://github.com/Azure/MachineLearningNotebooks/issues/1314).

Nit on `utils`  to make mypy happy.

## [0.33.1] - 2021-01-27

`HiveDb` class deprecared in favor of `HiveClient`.

## [0.34.0] - 2021-01-26

Deprecate from `utils` module:
- `read_yaml`
- `non_empty_dirs`
- `dict_to_namedtuple`
- `wrap_list_values_quotes`
- `get_fathers_mothers_kids_day`
- `get_friends_day`
- `is_special_day`
- `get_semi_month_pay_days`
- `get_obj_hash`
- `none_or_empty_pandas`
- `setup_logging`
- `in_clause_requirement`
- `split_on_letter`
- `template`
- `render_jinja_template`

`ipynb_utils` module deprecation. It will be removed from `muttlib` in version 1.0.0.
Some features will be migrated to `utils` module.

## [0.33.0] - 2021-01-20

New `plotting` module. An auxiliary toolkit for plotting that includes:

- `create_forecast_figure`: Plot trend, forecast and anomalies with history, anomaly and forecast phases.

## [0.32.0] - 2021-01-19

Refactored `BigQueryClient` client to support external clients as arguments and automatic closing of client when managed by itself.

Also complying with BaseClient interface.

## [0.31.0] - 2021-01-19
`ipynb_const` module deprecation. It will be removed from `muttlib` in version 1.0.0.

## [0.30.0] - 2021-01-19
`gcd` module deprecation. It will be removed from `muttlib` in version 1.0.0.

## [0.29.1] - 2020-12-29
Update README with PyPi install instructions.

## [0.29.0] - 2020-12-29
DBConn base classes refactor.

## [0.28.4] - 2020-12-29
Test that changelog has been modified in pipeline. This avoids merging MRs that have not updated the changelog.

## [0.28.3] - 2020-12-28
Pinned `pandas` version and upgraded `pyarrow` to version 2.0.0.

## [0.28.2] - 2020-12-22

CI/CD Pipelines updated, now [Merged Results](https://docs.gitlab.com/ee/ci/merge_request_pipelines/pipelines_for_merged_results) are activated.

Added to PR workflow explanation in CONTRIBUTING.md.

Closes #121 - Fix detached pipelines.

## [0.28.0] - 2020-12-10

Modify string filter `format_in_clause`, now supporting string values.
- Add tests to `utils` module.

## [0.27.10] - 2020-12-02

pylint setup final touches.

## [0.27.9] - 2020-12-02

Closes #85 - pylint `forecast.py`

## [0.27.8] - 2020-12-02

Closes #92 - pylint `file_processing.py`

## [0.27.0] - 2020-11-19

Closes #101 - Add tests to `ipynb_utils.py`:

- Add tests to `ipynb_utils` module.
- Remove plotting methods:
  - `top_categorical_vs_kdeplot`
  - `top_categorical_vs_heatmap`
  - `plot_agg_bar_charts`
  - `plot_category2category_pie_charts`
  - `plot_timeseries`

Closes #106 - Add --verbose option to twine upload

## [0.24.2] - 2020-10-24

Refactor to dbconn imports simpler.

## [0.23.0] - 2020-10-11

Refactor dbconn param parsing to use SQLAlchemy `make_url`.

## [0.16.0] - 2019-08-24

### Added

- This CHANGELOG file.

### Changed

- Use psycopg2-binary for postgres db connection.
- Give warnings about missing mysql and postgres modules.
