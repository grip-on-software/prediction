"""
File store interaction.

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

import argparse
from prediction.files import get_file_opener

def main():
    """
    Main entry point.
    """

    description = 'Upload or download files from stores'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', default='config.yml',
                        help='File to read store configuration from')
    parser.add_argument('--store', default='owncloud',
                        help='Store type to interact with')
    parser.add_argument('--upload', default=[], nargs='+',
                        help='Files to upload to the store')
    parser.add_argument('--download', default=[], nargs='+',
                        help='Files to download from the store')
    parser.add_argument('--remove', default=[], nargs='+',
                        help='Files to remove from the store')

    args = parser.parse_args()

    opener = get_file_opener(args)

    for upload in args.upload:
        with open(upload, 'r', encoding='utf-8') as local_file:
            with opener(upload, 'w') as upload_file:
                upload_file.write(local_file.read())

    for download in args.download:
        with open(download, 'w', encoding='utf-8') as local_file:
            with opener(download, 'r') as download_file:
                local_file.write(download_file.read())

    for remove in args.remove:
        opener.remove(remove)

if __name__ == "__main__":
    main()
