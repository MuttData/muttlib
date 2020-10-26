# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
