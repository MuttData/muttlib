# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
