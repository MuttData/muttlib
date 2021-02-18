from unittest.mock import patch, MagicMock

import pytest

from muttlib.dbconn import MongoClient


def test_init_without_host_or_seed_fails():
    with pytest.raises(ValueError, match=r".*[host|seed].*[host|seed].*"):
        MongoClient()
