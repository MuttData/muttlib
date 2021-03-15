"""This module provides a UNIX-ish interface to GDrive.

A minimal example of usage:

>>> root = GDrive(SERVICE_ACCOUNT_CREDS_JSON)
>>> root.ls()

TODO:
- owner attr
- return DriveNode objects
"""

import enum
import logging
import os
import pathlib

import httplib2
from oauth2client.service_account import ServiceAccountCredentials
import requests

logger = logging.getLogger(f'gdrive.{__name__}')


class GDriveMimeType(enum.Enum):
    """MimeTypes for Google Drive."""

    FOLDER = 'application/vnd.google-apps.folder'
    SPREADSHEET = 'application/vnd.google-apps.spreadsheet'


class GDriveURL(enum.Enum):
    """URLs from the Google Drive API."""

    FILES = "https://www.googleapis.com/drive/v3/files"
    DRIVES = "https://www.googleapis.com/drive/v3/drives"


SCOPE = [
    'https://spreadsheets.google.com/feeds',
    'https://www.googleapis.com/auth/drive',
]


class DriveClient:
    """Interface against Google Drive.

    Parameters
    ----------
    auth_file: str
        Filename for the json containing the service account credentials
    session: requests.Session
        Session to handle requests to the API
    """

    def __init__(self, auth_file=None, session=None):
        if session is None:
            session = requests.Session()
        self.session = session

        if auth_file is None:
            auth_file = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')

        if auth_file is not None:
            self.auth = ServiceAccountCredentials.from_json_keyfile_name(
                auth_file, SCOPE
            )
            self.login()

    def login(self):
        """Authorize client."""
        if not self.auth.access_token or (
            hasattr(self.auth, 'access_token_expired')
            and self.auth.access_token_expired
        ):
            http = httplib2.Http()
            self.auth.refresh(http)

        self.session.headers.update(
            {'Authorization': f'Bearer {self.auth.access_token}'}
        )

    def request(
        self,
        method,
        endpoint,
        params=None,
        data=None,
        json=None,
        files=None,
        headers=None,
    ):
        """Make a request and get the response.

        Parameters
        ----------
        method: str
            Either 'get' or 'post', the HTTP verb to be used
        endpoint: str
            The endpoint where to send the request to
        params: dict
            Params to be encoded on the URL query string
        data: dict
            Data to be included on the request body
        json: dict
            JSON data to be included on the request body
        files: str
            Files to be uploaded to the specified URL
        headers: dict
            Headers for the request
        """
        if method.upper() not in [
            'DELETE',
            'GET',
            'HEAD',
            'OPTIONS',
            'PATCH',
            'POST',
            'PUT',
        ]:
            raise ValueError(f"Invalid HTTP method: {method}")
        response = getattr(self.session, method)(
            endpoint, json=json, params=params, data=data, files=files, headers=headers
        )
        if not response.ok:
            try:
                logger.error(response.json()['error']['errors'])
            except KeyError:
                pass
            response.raise_for_status()
        return response

    def full_request(
        self,
        method,
        endpoint,
        key,
        params=None,
        data=None,
        json=None,
        files=None,
        headers=None,
    ):
        """Make a request and exhaust the resource.

        Parameters
        ----------
        method: str
            Either 'get', 'post' or 'delete'; the HTTP verb to be used
        endpoint: str
            The endpoint where to send the request to
        key: str
            The key from the data to construct the response from.
            This is the resource being exhausted.
        params: dict
            Params to be encoded on the URL query string
        data: dict
            Data to be included on the request body
        json: dict
            JSON data to be included on the request body
        files: str
            Files to be uploaded to the specified URL
        headers: dict
            Headers for the request

        """
        response_items = []
        if method.upper() not in [
            'DELETE',
            'GET',
            'HEAD',
            'OPTIONS',
            'PATCH',
            'POST',
            'PUT',
        ]:
            raise ValueError(f"Invalid HTTP method: {method}")
        response = getattr(self.session, method)(
            endpoint, json=json, params=params, data=data, files=files, headers=headers
        ).json()

        response_items.extend(response.get(key))

        page_token = response.get('nextPageToken', None)
        while page_token is not None:
            params['pageToken'] = page_token
            response = getattr(self.session, method)(
                endpoint,
                json=json,
                params=params,
                data=data,
                files=files,
                headers=headers,
            ).json()
            response_items.extend(response.get(key))
            page_token = response.get('nextPageToken', None)

        return response_items


class DriveNode:
    """Generic Drive __fs__ node.

    Parameters:
    -----------
        gid (str): id given by google drive to the resource
        name (str): humanly readable name for the resource
        parent (DriveNode): parent node in the hierarchy
    """

    def __init__(self, gid, name, parent):
        self._gid = gid
        self._name = name
        self._parent = parent

    @property
    def gid(self):
        """Return the Google Drive id for the node."""
        return self._gid

    @property
    def name(self):
        """Return the name for the node (might have collisions)."""
        return self._name

    @property
    def parent(self):
        """Return the parent object of this node."""
        return self._parent


class DriveFolder(DriveNode):
    """Interface to a Google Drive folder.

    Parameters
    ----------
    gid: str
        Google identifier for the folder or 'root'
    drive_client: DriveClient
        A google drive client that can be used to make
        requests to the API
    path: str
        A human readable path to alias the gid
    verbose: boolean
        If True logs relevant info
    """

    def __init__(
        self,
        gid,
        drive_client,
        path=None,
        parent=None,
        shared_drives=False,
        verbose=False,
    ):
        self.drive_client = drive_client
        self.verbose = verbose
        self.shared_drives = shared_drives

        r = self.drive_client.request(
            'get',
            f"{GDriveURL.FILES.value}/{gid}",
            params={'supportsAllDrives': self.shared_drives},
        ).json()

        super().__init__(gid, r.get('name'), parent)

        if path:
            self.path = path / self.name
        else:
            self.path = pathlib.PurePosixPath(gid)

        self._children = []
        self._init_node()

    def __truediv__(self, other):
        """Pathlib-like tree downwards navigation."""
        folders = set(
            x.get('id')
            for x in self._children
            if x.get('mimeType') == GDriveMimeType.FOLDER.value
        )
        if other not in folders:
            raise ValueError("other node not in this node or not a folder")
        return DriveFolder(
            other,
            self.drive_client,
            path=self.path,
            parent=self,
            verbose=self.verbose,
            shared_drives=self.shared_drives,
        )

    def _init_node(self):
        """Init the folder with it's children."""
        children = self.drive_client.full_request(
            'get',
            GDriveURL.FILES.value,
            'files',
            params={
                'q': f"trashed=False and ('{self.gid}' in parents)",
                'corpora': 'allDrives' if self.shared_drives else 'user',
                'includeItemsFromAllDrives': self.shared_drives,
                'supportsAllDrives': self.shared_drives,
            },
        )

        if str(self.path) in ('/', 'root'):
            # adding files shared with me to root if
            # - they have no parent
            # - shared drive folders
            # - shared drive root files
            other_files = []
            possible_parents = set()

            files_shared_with_me = self.drive_client.full_request(
                'get',
                GDriveURL.FILES.value,
                'files',
                params={'q': "trashed=False and sharedWithMe=True"},
            )

            other_files.extend(files_shared_with_me)

            if self.shared_drives:
                files_from_shared_drives = self.drive_client.full_request(
                    'get',
                    GDriveURL.FILES.value,
                    'files',
                    params={
                        'corpora': 'allDrives',
                        'includeItemsFromAllDrives': 'true',
                        'supportsAllDrives': 'true',
                        'q': 'trashed=False',
                    },
                )

                other_drives = set(
                    [
                        drive.get('id')
                        for drive in self.drive_client.full_request(
                            'get', GDriveURL.DRIVES.value, 'drives'
                        )
                    ]
                )

                possible_parents = possible_parents | other_drives

                other_files.extend(files_from_shared_drives)

            for other_file in other_files:
                r = self.drive_client.request(
                    'get',
                    f"{GDriveURL.FILES.value}/{other_file.get('id')}",
                    params={
                        'fields': "parents,id",
                        'supportsAllDrives': self.shared_drives,
                    },
                ).json()

                parents = r.get('parents')
                if (
                    parents is None
                    # or self.gid in parents
                    or len(set(parents) & possible_parents) > 0
                ):
                    children.append(other_file)

        self._children.extend(children)

    def ls(self):
        """List files on the folder."""
        if self.verbose:
            for _child in self._children:
                logger.info(
                    "%s (%s)\t\t%s",
                    _child.get('name'),
                    _child.get('id'),
                    _child.get('mimeType'),
                )
        return self._children

    def cd(self, other):
        """Change directory to a subfolder."""
        if other == "..":
            return self.parent
        return self / other

    def pwd(self):
        """Return current working directory."""
        if self.verbose:
            logger.info(self.path)
        return str(self.path)

    def touch(self, filename, mimetype):
        """Create a file of the specified mimetype on this folder.

        Parameters:
        -----------
            filename (str): the filename of the new file
            mimetype (str or GDriveMimeType): a valid mimetype
        """
        if isinstance(mimetype, GDriveMimeType):
            mimetype = mimetype.value

        r = self.drive_client.request(
            'post',
            GDriveURL.FILES.value,
            json={'parents': [self.gid], 'name': filename, 'mimeType': mimetype},
            params={'supportsAllDrives': self.shared_drives},
        )
        self._children.append(r.json())
        return r.json()

    def mkdir(self, foldername):
        """Create a new folder within the current working directory."""
        return self.touch(foldername, GDriveMimeType.FOLDER.value)

    def chown(
        self,
        email,
        role,
        permtype,
        item_gid=None,
        send_email=False,
        transfer_ownership=False,
    ):
        """Give/transfer permissions for a folder/file within the folder.

        Parameters
        ----------
        email: str
            Email of the transferee
        role: str
            Role of the transferee. Must be one of:
                - owner
                - organization
                - fileOrganizer
                - writer
                - commenter
                - reader
        permtype: str
            Type of permission to grant (user, group, domain, anyone)
        item_gid: str
            If the object to be transferred is not the folder, but a subitem, you
            can specify which with this parameter
        send_email: bool
            Whether a notification email should be sent
        transfer_ownership: bool
            Whether the permission granting should transfer ownership
        """
        valid_roles = (
            'owner',
            'organizer',
            'fileOrganizer',
            'writer',
            'commenter',
            'reader',
        )
        valid_permtypes = ('user', 'group', 'domain', 'anyone')

        if role not in valid_roles:
            raise ValueError(f"Unexpected role {role}. Valid roles are: {valid_roles}")

        if permtype not in valid_permtypes:
            raise ValueError(
                f"Unexpected permission {permtype}. Valid roles are: {valid_permtypes}"
            )

        endpoint = f"{GDriveURL.FILES.value}/{item_gid or self.gid}/permissions"
        params = dict(
            sendNotificationEmail=send_email, transferOwnership=transfer_ownership
        )
        body = dict(emailAddress=email, role=role, type=permtype)

        return self.drive_client.request(
            'post', endpoint, json=body, params=params
        ).json()

    def rm(self, gid):
        """Delete a node within this folder."""
        self.drive_client.request(
            'delete',
            f"{GDriveURL.FILES.value}/{gid}",
            params={'supportsAllDrives': self.shared_drives},
        )
        self._children = [x for x in self._children if x.get('id') != gid]
        return


def GDrive(creds_file, gid='root', verbose=False, shared_drives=False, session=None):
    """Entrypoint util to get a filesystem from root."""
    d = DriveClient(creds_file, session=session)
    return DriveFolder(gid, d, verbose=verbose, shared_drives=shared_drives)
