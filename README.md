# muttlib 🐶📚

[![pipeline status](https://gitlab.com/mutt_data/muttlib/badges/master/pipeline.svg)](https://gitlab.com/mutt_data/muttlib/-/commits/master)[![coverage report](https://gitlab.com/mutt_data/muttlib/badges/master/coverage.svg)](https://gitlab.com/mutt_data/muttlib/-/commits/master)

## Description

Library with helper code to start a data-related project.
By [Mutt Data](https://muttdata.ai/).

Current modules:

- `dbconn`: Somewhat homogeneus lib to access multiple DBs.
- `file_processing`: Helpers for concurrent file processing.
- `forecast`: Provides FBProphet a common interface to Sklearn and general
  utilities for forecasting problems, allowing wider and easier grid search for
  hyperparameters.
- `utils`: A single version of miscellaneous functions needed every now and then.
- `ipynb_const` and `ipynb_utils`: Utilities when doing exploratory work (helpful for jupyter notebooks).
- `gsheetsconn`: Module to make data interactions to/from Google Sheets <> Pandas.
- `gdrive`: Module that provides a UNIX-ish interface to GDrive.
- `gcd`: (Greatest Common Divisor, for lack of a better name - Ty @memeplex) Classes, abstract objects and other gimmicks.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Google Sheets Credentials](#google-sheets-credentials)
- [Testing](#testing)
- [Contributing](#contributing)
- [Credits](#contributing)
- [License](#license)

## Installation

To install base lib:
```bash
pip install git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib
```

Parquet and Feather support:
```bash
pip install git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[pyarrow]
```

IPython utils:
```bash
pip install git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[ipynb-utils]
```

Forecast:
```bash
pip install git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[forecast]
```

Misc DB support for dbconn:
```bash
pip install git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[oracle]
pip install git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[hive]
pip install git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[postgres]
pip install git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[mysql]
pip install git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[sqlserver]
pip install git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[moongo]
pip install git+https://gitlab.com/mutt_data/muttlib.git#egg=muttlib[ibis]
```

Install custom branch:
```bash
pip install -e git+https://gitlab.com/mutt_data/muttlib.git@AWESOME_FEATURE_BRANCH#egg=muttlib
```

## Usage
See the [documentation](https://mutt_data.gitlab.io/muttlib/) to get started with muttlib.

##  Google Sheets Credentials
To use the client in `gsheetsconn.py` one must first get the appropriate credentials in Json format. These are provided by GCP (Google's computing platform - cloud).

**Important note**: to obtain the necessary credentials, one must first have a valid GCP account which implies that billing for that account is already enabled. Having a standalone @gmail.com email will generally not suffice. This reference may probably help: [on billing and accounts support for GCP](https://cloud.google.com/support/billing/).

A good and simple step by step guide on how to get the Json file credentials can be seen in this [medium post](https://medium.com/@denisluiz/python-with-google-sheets-service-account-step-by-step-8f74c26ed28e). These credentials will be used by our client to read/write/edit files in Google Sheets.

The general idea is that in GCP we need to create or use an existing project, then enable the Sheets and Drive APIs for the selected project and finally create new service-account credentials for your script. Download them in Json format and put them somewhere accessible to your script.
There are other types of credentials but in this way we can have a server-side script that does not require explicit user consent to proceed with auth.

**Important note**: the service-account credentials will effectively provide us with a google-valid email, which will act as the "user" editing/modifying/etc. the data in Google Sheets.
This implies that this service email needs to have sufficient permissions to actually edit these files.
In general, giving permissions to the needed sheets will suffice.



## Contributing
We appreciate for considering to help out maintaining this project. If you'd like to contribute please read our [contributing guidelines](CONTRIBUTING.md).

## Credits

<!-- check-up -->

- Aldo Escobar
- Alejandro
- aoelvp94
- a-rusi
- Cristián Antuña
- CrossNox
- Eric Rishmuller
- fabian wolfmann
- Fabian Wolfmann
- Gabriel Miretti
- Javi M
- jdemonasterio
- josé
- Jose Castagnino
- Juan Martin Pampliega
- juapampliega
- Luis Alberto Hernandez
- Matías Battocchia
- mdm
- mutt-luis
- nox
- Pablo Andres Lorenzatto
- Pablo Lorenzatto
- Pedro Ferrari
- petobens
- risheric
- rishis07
- Santiago

## License
`muttlib` is licensed under the [Apache License 2.0](#https://gitlab.com/mutt_data/muttlib/-/blob/master/LICENCE).
