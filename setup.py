"""Muttlib setup file."""
import setuptools

import muttlib

with open('README.md', 'r', encoding='utf8') as fh:
    long_description = fh.read()

pyarrow_dep = ["pyarrow==0.13.0"]
holidays_dep = ['holidays>=0.9.5,<0.10.1']

setuptools.setup(
    name='muttlib',
    version=muttlib.__version__,
    author='Mutt Data',
    home_page="https://gitlab.com/mutt_data/muttlib/",
    keywords="data pandas spark data-analysis database data-munging",
    author_email='info@muttdata.ai',
    description='Collection of helper modules by Mutt Data.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    include_package_data=True,
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    setup_requires=['pytest-runner', 'wheel'],
    tests_require=["pytest", "pytest-cov", "pytest-html"] + pyarrow_dep + holidays_dep,
    test_suite='test',
    install_requires=[
        'jinja2',
        'pandas>=1.0.0',
        'progressbar2',
        'pyyaml',
        'scikit-learn',
        'scipy',
        'sqlalchemy',
    ],
    extras_require={
        'pyarrow': pyarrow_dep,
        'oracle': ['cx_Oracle'],
        'hive': ['pyhive>=0.6.1'],
        'postgres': ['psycopg2-binary'],
        'mysql': ['pymysql'],
        'sqlserver': ['pymssql'],
        'mongo': ['pymongo'],
        'ibis': ['ibis', 'ibis-framework[impala]', 'impyla'] + pyarrow_dep,
        'ipynb-utils': [
            'IPython',
            'jinja2',
            'jinjasql',
            'matplotlib',
            'numpy',
            'pandas',
            'seaborn',
            'tabulate',
            'textwrap3',
        ],
        'dev': [
            'pre-commit',
            'flake8',
            'flake8-bugbear',
            'flake8-docstrings',
            'black',
            'mypy',
            'pylint',
            'bump',
            'sphinx',
            'sphinx_rtd_theme',
            'm2r @ git+https://github.com/crossnox/m2r@dev#egg=m2r',
        ],
        'forecast': ['fbprophet'] + holidays_dep,
        'gsheets': ['gspread_pandas'],
    },
)
