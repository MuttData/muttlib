'''Muttlib setup file.'''
import setuptools

import muttlib

with open('README.md', 'r', encoding='utf8') as fh:
    long_description = fh.read()

pyarrow_dep = ['pyarrow==2.0.0']
holidays_dep = ['holidays>=0.10.2']

#  define 'extra_dependencies'
extra_dependencies = {
    'pyarrow': pyarrow_dep,
    'oracle': ['cx_Oracle'],
    'hive': ['pyhive>=0.6.1', 'thrift'],
    'postgres': ['psycopg2-binary'],
    'mysql': ['pymysql'],
    'sqlserver': ['pymssql'],
    'mongo': ['pymongo'],
    'teradata': ['teradatasql==17.0.0.8'],
    'bigquery': ['google-cloud-bigquery'],
    'ibis': ['ibis-framework', 'ibis-framework[impala]', 'impyla'] + pyarrow_dep,
    'gdrive': ['oauth2client', 'requests'],
    'dev': [
        'flake8-bugbear',
        'flake8-docstrings',
        'bump2version',
        'sphinx==3.2.1',
        'sphinx_rtd_theme',
        'm2r2',
        'betamax',
        'betamax-serializers',
        'pre-commit==2.2.0',
        'interrogate',
        'isort==4.3.21',
        'black==19.10b0',
        'mypy==0.770',
        'flake8==3.7.8',
        'pylint==2.4.4',
        'nox',
    ],
    'test': [
        'freezegun',
        'interrogate',
        'nox',
        'pytest',
        'pytest-xdist',
        'pytest-cov',
        'pytest-html',
        'pytest-mpl==0.12',
        'betamax',
        'betamax-serializers',
    ]
    + pyarrow_dep
    + holidays_dep,
    'forecast': ['fbprophet', 'pystan==2.19.1.1'] + holidays_dep,
    'gsheets': ['gspread_pandas'],
}

# create 'all' extras
all_extras = []
for extra_dep in extra_dependencies.values():
    if extra_dep not in all_extras:
        all_extras += extra_dep
extra_dependencies.update({'all': all_extras})

setuptools.setup(
    name='muttlib',
    version=muttlib.__version__,
    author='Mutt Data',
    home_page='https://gitlab.com/mutt_data/muttlib/',
    keywords='data pandas spark data-analysis database data-munging',
    author_email='info@muttdata.ai',
    description='Collection of helper modules by Mutt Data.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    include_package_data=True,
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    setup_requires=['pytest-runner', 'wheel'],
    tests_require=["pytest", "pytest-cov", "pytest-html", "betamax"]
    + pyarrow_dep
    + holidays_dep,
    test_suite='test',
    install_requires=[
        'deprecated',
        'jinja2',
        'pandas>=1.0.0',
        'progressbar2',
        'pyyaml',
        'scikit-learn',
        'scipy',
        "sqlalchemy<1.4.0,>=1.3.0",
        'numpy==1.19.5',
        'jinjasql',
        'IPython',
        'matplotlib',
    ],
    extras_require=extra_dependencies,
)
