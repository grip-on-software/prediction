"""
TensorFlow ARFF dataset loader.
"""

import logging
import tensorflow as tf
from scipy.io import arff
import numpy as np
from sklearn.model_selection import train_test_split

class Dataset(object):
    """
    Dataset selector and provider.
    """

    TRAIN = 0
    TEST = 1

    def __init__(self, args):
        self.args = args
        self.load_datasets()
        self._batches = {}

    def _translate(self, indices, translation):
        for index in indices:
            try:
                yield int(index)
            except ValueError:
                if index in translation:
                    yield translation[index]
                elif ',' in index:
                    self._translate(index.split(','), translation)
                else:
                    raise ValueError('Index {0} could not be understood'.format(index))

    def _get_labels(self, data, translation):
        label_index = next(self._translate([self.args.label], translation))
        column = np.nan_to_num(data[:, label_index])
        labels = column.astype(int)
        if np.any(column - labels) != 0:
            raise ValueError('Label column {0} has non-round numbers'.format(self.args.label))

        if self.args.binary is not None:
            labels = (labels >= self.args.binary).astype(int)

        return labels, label_index

    def _select_data(self, full_data, meta):
        name_translation = dict(zip(meta.names(), range(full_data.shape[1])))

        if self.args.index:
            indexes = set(self._translate(self.args.index, name_translation))
        else:
            indexes = set(range(full_data.shape[1]))

        indexes.remove(0)
        if self.args.remove:
            indexes -= set(self._translate(self.args.remove, name_translation))

        if self.args.label:
            labels, label_index = self._get_labels(full_data, name_translation)
            indexes.remove(label_index)

            logging.debug('Selected label %d: %r', label_index, labels)
        else:
            labels = np.random.randint(0, 1+1, size=(full_data.shape[0],))

        logging.debug('Leftover indices: %r', indexes)
        return indexes, labels

    def load_datasets(self):
        """
        Load the dataset and split into train/test, and inputs/labels.
        """

        with open(self.args.filename) as features_file:
            data, meta = arff.loadarff(features_file)

        logging.debug('Metadata:\n%r', meta)

        full_data = np.array([[cell for cell in row] for row in data],
                             dtype=np.float32)
        indexes, labels = self._select_data(full_data, meta)

        dataset = np.nan_to_num(full_data[:, tuple(indexes)])
        names = [name for index, name in enumerate(meta) if index in indexes]

        logging.debug('Leftover column names: %r', names)

        train_data, test_data, train_labels, test_labels = \
            train_test_split(dataset, labels, test_size=self.args.test_size)

        self.data_sets = {
            self.TRAIN: (train_data, train_labels),
            self.TEST: (test_data, test_labels)
        }
        self.num_features = train_data.shape[1]
        self.num_labels = max(labels)+1

    def get_batches(self, data_set):
        """
        Create batches of the input/labels.
        """

        if data_set in self._batches:
            return self._batches[data_set]
        if data_set not in self.data_sets:
            raise IndexError('Data set #{} does not exist'.format(data_set))

        with tf.name_scope('input'):
            # Input data, pin to CPU because rest of pipeline is CPU-only
            with tf.device('/cpu:0'):
                inputs = tf.constant(self.data_sets[data_set][0])
                labels = tf.constant(self.data_sets[data_set][1])

                inputs, labels = tf.train.slice_input_producer([inputs, labels],
                                                               num_epochs=self.args.num_epochs)
                inputs, labels = tf.train.batch([inputs, labels],
                                                batch_size=self.args.batch_size,
                                                num_threads=self.args.num_threads,
                                                allow_smaller_final_batch=True)

        self._batches[data_set] = [inputs, labels]
        return self._batches[data_set]

    def clear_batches(self, data_set):
        """
        Remove cached batches.
        """

        if data_set in self._batches:
            del self._batches[data_set]
