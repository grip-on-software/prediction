"""
TensorFlow checkpoint cleaner.
"""

import copy
import glob
import os

class Cleaner(object):
    """
    File cleaner for TensorFlow checkpoint files.
    """

    DEFAULT_PATTERNS = [
        "checkpoint", "graph.phtxt", "events.out.tfevents.*", "model.ckpt-*.*",
        "eval/events.out.tfevents.*"
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
