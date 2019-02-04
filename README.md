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
pip install -e git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib
```

IPython utils:
```
pip install -e git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[ipynb-utils]
```

Misc DB suppoort for dbconn:
```
pip install -e git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[oracle]
pip install -e git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[hive]
pip install -e git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[postgres]
pip install -e git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[sqlserver]
pip install -e git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[moongo]
pip install -e git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[ibis]
```

# Dirty Dry-run (done dirt cheap)
```
pip install -e git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib
python -c 'from muttlib import dbconn, utils'

pip install -e git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[ipynb-utils]
python -c 'from muttlib import ipynb_const, ipynb_utils'
```
# Testing
```
pytest -v test/utils_test.py
```
