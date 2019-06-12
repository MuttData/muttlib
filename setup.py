"""Muttlib setup file."""
import setuptools
from setuptools.dist import Distribution

import muttlib

with open('README.md', 'r', encoding='utf8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='muttlib',
    version=muttlib.__version__,
    author='Mutt Data',
    home_page = "https://gitlab.com/mutt_data/muttlib/",
    keywords = "data pandas spark data-analysis database data-mungingz",
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
    tests_require=["pytest", "pytest-cov", "pytest-html"],
    test_suite='test',
    install_requires=[
        'jinja2',
        'pandas',
        'progressbar2',
        'pyarrow',
        'pyyaml',
        'sqlalchemy',
        'scipy',
    ],
    extras_require={
        'oracle': ['cx_Oracle'],
        'hive': ['pyhive'],
        'postgres': ['psycopg2'],
        'sqlserver': ['pymssql'],
        'mongo': ['pymongo'],
        'ibis': ['ibis'],
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
        ],
    },
)
