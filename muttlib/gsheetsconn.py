"""
Google Sheets <> Pandas connector

Remember it is crucial that you acquire correct Google Cloud Platform credentials
to interact with GSheets and GDrive APIs with authorization. This come in the form of a
json file whose path needs to be passed to this during initialization.
See the README.md for more details on how to acquire one.

A full round trip of how this module operates scan be seen as follows:

> import pandas as pd
> ​from muttlib import gsheetsconn
>
> GOOGLE_SHEETS_SECRETS_JSON_FP = "/some/local/dir/path/to/json/file.json"
> # A spreadhseet id is part of the url used in your browser
> GSHEETS_SPREAD_ID = "1psc7zc-yppfpmEUWAi6hW4GLstiPyZqlov6w6eATeuU"
> GSHEETS_WORKSHEET_NAME = 'test'
>
> gsheets_client = GSheetsClient(GOOGLE_SHEETS_SECRETS_JSON_FP)
>​
> # Construct random data from integers
> keys = list('ABC')
> numrows = 100
> str_length = 4
> df = pd.DataFrame(
>     pd.np.random.randint(0, 15, sizimport pandas as pd
> # And letters
> df['letters'] = pd.util.testing.rands_array(str_length, numrows)
​>
> # Test push working gsheets conn
> spreadsheet_id, worksheet = (GSHEETS_SPREAD_ID, GSHEETS_WORKSHEET_NAME)
​>
> gsheets_client.insert_from_frame(df, spreadsheet_id, index=True, worksheet=worksheet)
>​
> return_df = gsheets_client.to_frame(spreadsheet_id, worksheet=worksheet)

Google Sheets Credentials

To use the client in gsheetsconn.py one must first get the appropriate
credentials in Json format. These are provided by GCP (Google's computing platform - cloud).

Important note: to obtain the necessary credentials, one must first have a valid GCP account
which implies that billing for that account is already enabled. Having a standalone @gmail.com email
will generally not suffice. This reference may probably help: on billing and accounts support
for GCP.

A good and simple step by step guide on how to get the Json file credentials can be seen
in this medium post. These credentials will be used by our client to read/write/edit files
in Google Sheets.

The general idea is that in GCP we need to create or use an existing project, then enable
the Sheets and Drive APIs for the selected project and finally create new service-account
credentials for your script. Download them in Json format and put them somewhere accessible
to your script.

There are other types of credentials but in this way we can have a server-side script that does
not require explicit user consent to proceed with auth.

Important note: the service-account credentials will effectively provide us with a google-valid
email, which will act as the "user" editing/modifying/etc. the data in Google Sheets.

This implies that this service email needs to have sufficient permissions to actually
edit these files. In general, giving permissions to the needed sheets will suffice.
"""
# TODO July 29, 2019: extend docstring with examples

import logging
from pathlib import Path
import re
from string import ascii_letters
from typing import Any, Dict, List

logger = logging.getLogger(f'gsheetsconn.{__name__}')  # NOQA

try:
    import gspread_pandas  # noqa: F401 # pylint: disable=unused-import
except ModuleNotFoundError:
    logger.warning("No GSpread support.")

from gspread_pandas import Spread, conf as gconf


class GSheetsClient:
    """Create Google Sheets client to interact with the app."""

    VALID_CELL_IDS_PATTERN = r"^[A-Z]+\d+$"
    COMPILED_VALID_CELL_IDS_PATTERN = re.compile(VALID_CELL_IDS_PATTERN)

    def __init__(
        self,
        conf_filepath: Path = None,
        user: str = None,
        auth_scope: List[str] = None,
        auth_creds: Any = None,
        conf: Dict = None,
    ) -> None:

        self.conf_filepath = conf_filepath or Path()
        self.user = user
        self.auth_scope = auth_scope
        self.auth_creds = auth_creds
        self.conf = conf or gconf.get_config(
            conf_dir=self.conf_filepath.parent, file_name=self.conf_filepath.name
        )

    def _get_auth(self):
        """Create valid OAuth2 credentials for gsheets."""
        logger.debug(f"Getting oauth2 credentials from {self.conf_filepath}.")
        if not self.auth_creds:
            if self.auth_scope is None:
                self.auth_creds = gconf.get_creds(config=self.conf, user=self.user)
            else:
                self.auth_creds = gconf.get_creds(
                    config=self.conf, scope=self.auth_scope, user=self.user
                )
        return self.auth_creds

    def get_spreadsheet(
        self,
        id_or_name,
        worksheet=None,
        create_spread=True,
        create_sheet=True,
        **kwargs,
    ):
        """Get Spreadsheet object.

        Parameters
        ---------
        id_or_name : str
            Spreadsheet id or name as in Drive.
        worksheet : str or int or Optional
            Name or index of Worksheet.

        Notes
        -----
            A spreadsheet is the whole document, whilst a worksheet is a single
            sheet/tab in google Sheets.
            Spread name needs not already exist, it'll be created by your
            user if missing.
        """
        auth_creds = self._get_auth()
        logger.debug(f"Loading spreadsheet '{id_or_name}'.")
        spread = Spread(
            spread=id_or_name,
            sheet=worksheet,
            create_spread=create_spread,
            create_sheet=create_sheet,
            creds=auth_creds,
            **kwargs,
        )
        return spread

    def _sheets_col2num(self, col):
        """
        Convert column-letter notation to ix.
        Ref:
            https://stackoverflow.com/questions/7261936/convert-an-excel-or-spreadsheet-column-letter-to-its-number-in-pythonic-fashion
        """
        num = 0
        for c in col:
            if c in ascii_letters:
                num = num * 26 + (ord(c.upper()) - ord('A')) + 1
        return num

    def _is_valid_cell_loc(self, val):
        """Checks correct GSheets cell location format."""
        rv = self.COMPILED_VALID_CELL_IDS_PATTERN.match(val)
        if not rv:
            raise ValueError(
                f"Value passed was `{val}` which is not a valid type of cell location for GSheets."
            )

    def to_frame(
        self,
        spreadsheet,
        worksheet=None,
        first_cell_loc=None,
        index_col=None,
        first_row=1,
        num_header_rows=1,
        create_spread=False,
        create_sheet=False,
        **kwargs,
    ):
        """
        Get data from spreadsheet and convert to DF.

        Parameters
        ----------
        spreadsheet : str
            Spreadsheet id or name as in Drive.
        worksheet : str or int or none
            Name or index of Worksheet.
        first_cell_loc : str or None
            Gsheet-like cell location of top-left-most cell in the data.
            This defines and overrides the `index_col` and `first_row` args.
        index_col : int or str or None
            Number of col to be set as index.
        first_row : int or None
            Number of row at which data, including header, starts.
        num_header_rows : int or None
            Number of rows to be read as header.
        """
        if index_col is not None and not (
            isinstance(index_col, str) or isinstance(index_col, int)
        ):
            raise ValueError(
                f"Valid types for `index_col` are `str` or `int`. Value passed was `{index_col}`."
            )
        spread = self.get_spreadsheet(
            spreadsheet,
            worksheet,
            create_spread=create_spread,
            create_sheet=create_sheet,
        )
        if first_cell_loc:
            first_cell_loc = first_cell_loc.upper()
            self._is_valid_cell_loc(first_cell_loc)
            index_col, first_row = tuple(
                filter(None, re.split(r'([aA-zZ]+)', first_cell_loc))
            )
            first_row = int(first_row)

        if isinstance(index_col, str):
            index_col = self._sheets_col2num(index_col)

        logger.debug(
            f"Pulling data to a DF, from spreadsheet '{spreadsheet}', worksheet `{worksheet}`."
        )
        df = spread.sheet_to_df(
            index=index_col,
            header_rows=num_header_rows,
            start_row=first_row,
            sheet=worksheet,
            **kwargs,
        )
        if (
            first_cell_loc
        ):  # by default reset-indexes when the first-cell-loc was passed
            df.reset_index(inplace=True)

        return df

    def insert_from_frame(
        self,
        df,
        spreadsheet,
        index=False,
        header=True,
        first_cell_loc='A1',
        worksheet=None,
        preclean_sheet=True,
        null_fill_value='',
        freeze_headers=True,
        **kwargs,
    ):
        """
        Convert DF to spreadsheet data.

        Parameters
        ----------
        df : pd.DataFrame
            The data to upload.
        spreadsheet : str
            Spreadsheet id or name as in Drive.
        index : bool
            Upload DF index.
        header : bool)
            Upload DF header.
        first_cell_loc : int or Nonee
            Exact row/column location to start dumping the data.
        worksheet : str or int or None
            Name or index of Worksheet.
        preclean_sheet : int or None
            To clean previous data in worksheet or not.
        null_fill_value : str or None
            The value representing nulls in data.
        """
        spread = self.get_spreadsheet(spreadsheet, worksheet)
        spread.df_to_sheet(
            df,
            sheet=worksheet,
            index=index,
            headers=header,
            start=first_cell_loc,
            replace=preclean_sheet,
            freeze_headers=freeze_headers,
            fill_value=null_fill_value,
            **kwargs,
        )
        logger.info(
            f"Inserted {df.size} values of a {df.shape} DataFrame into "
            f"spreadsheet '{spreadsheet}'."
        )
