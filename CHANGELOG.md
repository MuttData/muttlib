# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2021-02-24

Refactor MongoClient to match new dbconn structure.

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
