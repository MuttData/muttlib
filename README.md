# muttlib 🐶📚

Library with helper code to start a project.

Current modules:
- `dbconn`: Somewhat homogeneus lib to access multiple DBs.
- `file_processing`: Helpers for concurrent file processing.
- `utils`: A single version of miscellaneous functions needed every now and then.
- `ipynb_const.py` and `ipynb_utils.py`: Utilities when for exploratory work.


# Install
Base lib:
```
pip install git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib
```

IPython utils:
```
pip install git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[ipynb-utils]
```

Misc DB support for dbconn:
```
pip install git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[oracle]
pip install git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[hive]
pip install git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[postgres]
pip install git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[sqlserver]
pip install git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[moongo]
pip install git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[ibis]
```

# Dirty Dry-run (done dirt cheap)
```
pip install -e git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib
python -c 'from muttlib import dbconn, utils'

pip install -e git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[ipynb-utils]
python -c 'from muttlib import ipynb_const, ipynb_utils'
```

# Pre-Commit for Version Control Integration

When developing you'll have to use [pre-commit](https://pre-commit.com/) to run a series
of linters and formatters, defined in `.pre-commit-config.yaml`, on each staged file.
There are two ways to install to these binaries:

## Global install of binaries

The easiest way to set this up is by first installing `pipx` with
```commandline
pip install --user pipx
```
and then use `pipx` to actually install the `pre-commit` binary along the linters and
formatters globally:
```commandline
pipx install pre-commit --verbose
pipx install isort --spec git+https://github.com/timothycrosley/isort@develop --verbose
pipx install flake8 --spec git+https://github.com/PyCQA/flake8 --verbose
pipx inject flake8 flake8-bugbear flake8-docstrings --verbose
pipx install black --verbose
pipx install mypy --verbose
pipx install pylint --verbose
```

Once that's done then simply run `pre-commit install` and you are good to go: every time
you do a `git commit` it will run the `pre-commit` hooks defined in
`.pre-commit-config.yaml`.

## Local install of binaries

The binaries are also listed as `dev` packages in `setup.py`. Therefore you can
alternatively install `muttlib` locally in a virtual environment using `pipenv`. To do
that first clone the repo and then run

```commandline
pipenv install -e .[dev] --skip-lock
```

Since the `.pre-commit-config.yaml` forces `pre-commit` to execute the environment of the
shell at the time of `git commit` you'll then have to run `git commit` from within a
`pipenv` subshell by first running `pipenv shell`.

