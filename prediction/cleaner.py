"""
TensorFlow checkpoint cleaner.

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

import copy
import glob
import os

class Cleaner:
    """
    File cleaner for TensorFlow checkpoint files.
    """

    DEFAULT_PATTERNS = [
        "checkpoint", "graph.phtxt", "events.out.tfevents.*", "model.ckpt-*.*",
        "eval/events.out.tfevents.*", "net.*.hdf5"
    ]

    def __init__(self, patterns=None):
        if patterns is None:
            patterns = self.DEFAULT_PATTERNS

        self.set_patterns(patterns)

    def add_pattern(self, pattern):
        """
        Add a glob pattern to the list of file patterns to match and remove.
        """

        self._patterns.append(pattern)

    def set_patterns(self, patterns):
        """
        Relace the list of patterns to match and remove.
        """

        self._patterns = copy.copy(patterns)

    def clean(self, directory):
        """
        Remove checkpoint files and related files from the `directory`.
        """

        for pattern in self._patterns:
            for path in glob.glob(os.path.join(directory, pattern)):
                os.remove(path)
