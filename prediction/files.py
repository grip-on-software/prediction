"""
Abstraction layer for file reading and writing.

We support local file reads as well as ownCloud communication.
"""

from io import BytesIO
import keyring
import owncloud
import yaml

def get_file_opener(args):
    """
    Select an appropriate function or class to open files from the store defined
    by the arguments.
    """

    if args.store == 'owncloud':
        return OwnCloudFile
    if args.store == 'local':
        return open

    raise ValueError('Invalid file opener')

class OwnCloudFile(object):
    """
    Handler for opening a file on ownCloud
    """

    _config = None

    def __init__(self, path, mode='r'):
        self._path = path
        self._mode = mode

        if self._config is None:
            with open('config.yml') as config_file:
                self._config = yaml.load(config_file).get('owncloud')

        self._client = owncloud.Client(self._config['url'],
                                       verify_certs=self._config.get('verify'))

        username = self._config['username']
        if self._config.get('keychain'):
            password = keyring.get_password('owncloud', username)
        else:
            password = self._config['password']

        self._client.login(username, password)
        self._stream = None

    @property
    def path(self):
        """
        Retrieve the path name of the file.
        """

        return self._path

    @property
    def mode(self):
        """
        Retrieve the open mode of the file.
        """

        return self._mode

    def __enter__(self):
        if self._mode == 'w':
            self._stream = BytesIO()
        else:
            self._stream = BytesIO(self._client.get_file_contents(self._path))

        return self._stream

    def __exit__(self, exc_type, exc_value, traceback):
        if self._mode == 'w':
            self._client.put_file_contents(self._path, self._stream.getvalue())

        self._stream.close()
