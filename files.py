"""
File store interaction.
"""

from __future__ import print_function
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
        with open(upload, 'r') as local_file:
            with opener(upload, 'w') as upload_file:
                upload_file.write(local_file.read())

    for download in args.download:
        with open(download, 'w') as local_file:
            with opener(download, 'r') as download_file:
                local_file.write(download_file.read())

    for remove in args.remove:
        opener.remove(remove)

if __name__ == "__main__":
    main()
