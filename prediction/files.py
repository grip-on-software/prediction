"""
Abstraction layer for file reading and writing.

We support local file reads as well as ownCloud communication.

Copyright 2017-2020 ICTU
Copyright 2017-2022 Leiden University
Copyright 2017-2023 Leon Helwerda

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from io import StringIO
import keyring
import owncloud
import yaml

def get_file_opener(args):
    """
    Select an appropriate function or class to open files from the store defined
    by the arguments.
    """

    if args.store == 'owncloud':
        OwnCloudFile.get_config(args.config)
        return OwnCloudFile
    if args.store == 'local':
        return open

    raise ValueError('Invalid file opener')

class OwnCloudFile:
    """
    Handler for opening a file on ownCloud
    """

    _config = None
    _client = None

    @classmethod
    def get_config(cls, config_filename='config.yml'):
        """
        Load configuration for ownCloud from the provided configuration
        filename.
        """

        if cls._config is None:
            with open(config_filename, encoding='utf-8') as config_file:
                cls._config = yaml.safe_load(config_file).get('owncloud')

        return cls._config

    @classmethod
    def clear_config(cls):
        """
        Clear the currently loaded configuration, if any, such as to allow
        loading from a different configuration file.
        """

        cls._config = None
        cls._client = None

    @classmethod
    def _get_client(cls):
        """
        Retrieve the ownCloud client.
        """

        if cls._client is None:
            config = cls.get_config()
            cls._client = owncloud.Client(config.get('url'),
                                          verify_certs=config.get('verify'))

            username = config.get('username')
            if config.get('keychain'):
                password = keyring.get_password('owncloud', username)
            else:
                password = config.get('password')

            cls._client.login(username, password)

        return cls._client

    @classmethod
    def remove(cls, path):
        """
        Remove a path from the ownCloud store.
        """

        client = cls._get_client()
        client.delete(path)

    def __init__(self, path, mode='r'):
        self._path = path
        self._mode = mode

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
            self._stream = StringIO()
        else:
            contents = self._get_client().get_file_contents(self._path)
            self._stream = StringIO(contents.decode("utf-8"))

        return self._stream

    def __exit__(self, exc_type, exc_value, traceback):
        if self._mode == 'w':
            contents = self._stream.getvalue().encode("utf-8")
            self._get_client().put_file_contents(self._path, contents)

        self._stream.close()
