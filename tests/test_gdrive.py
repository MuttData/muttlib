"""muttlib.gdrive test suite."""
import os
import unittest

import betamax
from betamax import Betamax
from betamax_serializers import pretty_json
import requests

from muttlib.gdrive import GDrive, GDriveMimeType

betamax.Betamax.register_serializer(pretty_json.PrettyJSONSerializer)

with Betamax.configure() as config:
    config.cassette_library_dir = 'tests/cassettes'
    # https://betamax.readthedocs.io/en/latest/record_modes.html
    record_mode = os.environ.get('MUTTLIB_RECORD_MODE', 'once')
    config.default_cassette_options['record_mode'] = record_mode
    config.default_cassette_options['serialize_with'] = 'prettyjson'

# TODO: provide a reasonable default
CREDS_FILE = os.getenv('MUTTLIB_GOOGLE_TEST_SERVICE_ACCOUNT_CREDS')


class BetamaxTest(unittest.TestCase):
    """Base TestCase class for using Betamax.

    This implementation is based on betamax.fixtures.unittest.py.
    """

    @classmethod
    def setUpClass(cls):
        """Set up the class.

        Sets a session and recorder to be used across all tests.
        """
        cls.session = requests.Session()
        cls.recorder = Betamax(session=cls.session)

    def generate_cassette_name(self):
        """Generate the cassette name for the current test."""
        return f'{self.__class__.__name__}.{self._testMethodName}'

    def setUp(self):
        """Set the recorder and cassette for the current test."""
        super(BetamaxTest, self).setUp()
        self.recorder = self.__class__.recorder
        cassette_name = self.generate_cassette_name()

        self.recorder.use_cassette(cassette_name)
        self.recorder.start()

    def tearDown(self):
        """Tear down the test.

        Stops the recorder, takes the cassette out, grabs a pen and
        rewinds the cassette for future playback.
        """
        super(BetamaxTest, self).tearDown()
        self.recorder.stop()


class TestGDriveRoot(BetamaxTest):
    """Test the GDrive from root."""

    @classmethod
    def setUpClass(cls):
        """Set the GDrive to be used accross all this class' tests."""
        super(TestGDriveRoot, cls).setUpClass()
        with cls.recorder.use_cassette(f'{cls.__name__}'):
            cls.gdrive = GDrive(CREDS_FILE, gid='root', session=cls.session)

    def setUp(self):
        """Set the test."""
        super(TestGDriveRoot, self).setUp()
        self.gdrive = TestGDriveRoot.gdrive

    def test_parent(self):
        """Test the parent of root is None."""
        assert self.gdrive.parent is None

    def test_path(self):
        """Test the path for root is correct."""
        assert str(self.gdrive.path) == 'root'

    def test_ls(self):
        """Test the contents of the folder."""
        root_children = self.gdrive.ls()
        assert root_children is not None

        for child in root_children:
            assert 'id' in child

    def test_pwd(self):
        """Test the pwd method."""
        assert self.gdrive.pwd() == 'root'

    def test_touch_and_rm(self):
        """Test creating and deleting a file."""
        r = self.gdrive.touch("test_spreadsheet", GDriveMimeType.SPREADSHEET.value)

        assert 'id' in r

        children = self.gdrive.ls()

        assert r.get('id') in set(x.get('id') for x in children)

        self.gdrive.rm(r.get('id'))

        children = self.gdrive.ls()

        assert r.get('id') not in set(x.get('id') for x in children)
