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

Misc DB suppoort for dbconn:
```
pip install git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[oracle]
pip install git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[hive]
pip install git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[postgres]
pip install git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[sqlserver]
pip install git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[moongo]
pip install git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[ibis]
```

Install custom branch:
```
pip install -e git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib@AWESOME_FEATURE_BRANCH
```

# Testing
Run all tests:
```
python setup.py test
```
Note: Some extra deps might be needed. Those can be added with this `pip install -e .[ipynb-utils]`.

Run coverage:
```
py.test --cov-report html:cov_html --tb=short -q --cov-report term-missing --cov=. test/
```

That should output a short summary and generate a dir `cov_html/` with a detailed HTML report that can be viewed by opening `index.html` in your browser.


## Dirty Dry-run (done dirt cheap)
```
pip install git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib
python -c 'from muttlib import dbconn, utils'

pip install git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[ipynb-utils]
python -c 'from muttlib import ipynb_const, ipynb_utils'
```
