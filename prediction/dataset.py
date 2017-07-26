"""
TensorFlow ARFF dataset loader.
"""

import logging
import tensorflow as tf
from scipy.io import arff
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class Dataset(object):
    """
    Dataset selector and provider.
    """

    # Indexes of the prepurposed data sets in the data_sets variable
    TRAIN = 0
    TEST = 1
    VALIDATION = 2

    # Indexes of the inputs, labels and other column data in each data set
    INPUTS = 0
    LABELS = 1
    WEIGHTS = 2
    WEATHER = 3

    # Indexes in the original dataset of uniquely identifying keys
    PROJECT_KEY = 0
    SPRINT_KEY = 1

    def __init__(self, args):
        self.args = args
        self.load_datasets()
        self._batches = {}
        self.ratios = None

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

    def _load(self):
        with open(self.args.filename) as features_file:
            data, meta = arff.loadarff(features_file)

        logging.debug('Metadata:\n%r', meta)

        full_data = np.array([[cell for cell in row] for row in data],
                             dtype=np.float32)

        return full_data, meta

    def _select_data(self, full_data, meta):
        name_translation = dict(zip(meta.names(), range(full_data.shape[1])))

        if self.args.index:
            indexes = set(self._translate(self.args.index, name_translation))
        else:
            indexes = set(range(full_data.shape[1]))

        indexes.discard(self.PROJECT_KEY)
        if self.args.remove:
            indexes -= set(self._translate(self.args.remove, name_translation))

        if self.args.label:
            labels, label_index = self._get_labels(full_data, name_translation)
            indexes.discard(label_index)

            logging.debug('Selected label %d: %r', label_index, labels)
        else:
            labels = np.random.randint(0, 1+1, size=(full_data.shape[0],))

        names = [name for index, name in enumerate(meta) if index in indexes]
        logging.debug('Leftover indices: %r', indexes)
        logging.debug('Leftover column names: %r', names)

        dataset = np.nan_to_num(full_data[:, tuple(indexes)])

        projects = full_data[:, self.PROJECT_KEY]
        project_splits = np.squeeze(np.argwhere(np.diff(projects) != 0) + 1)
        return dataset, labels, project_splits, full_data

    @classmethod
    def _last_sprint_weather(cls, labels, project_splits):
        project_labels = np.split(labels, project_splits)
        return np.hstack([
            np.hstack([np.nan, np.roll(pl, 1, axis=0)[1:]]) for pl in project_labels
        ])

    @classmethod
    def _last_sprint_weather_accuracy(cls, labels, project_splits=None,
                                      weather=None, name='data'):
        if weather is None:
            weather = cls._last_sprint_weather(labels, project_splits)

        num_correct = np.count_nonzero(np.equal(labels, weather))
        if project_splits is None:
            num_unclassifiable = 0
        else:
            num_unclassifiable = len(project_splits)

        accuracy = (num_correct - num_unclassifiable) / float(len(labels))
        logging.info('Last sprint weather accuracy (%s): %.2f', name, accuracy)
        return weather, num_correct, accuracy

    def _roll_sprints(self, project_splits, dataset, labels, weather):
        # Roll the sprints such that a sprint has features from the sprint
        # before it, while the labels remain the same for that sprint.
        # Remove the sprint at the start of a project in lack of features.
        logging.debug('Project splits: %r', project_splits)

        latest_indexes = np.hstack([project_splits-1, -1])
        latest = dataset[latest_indexes, :]
        latest_labels = labels[latest_indexes]

        if self.args.roll_labels:
            dataset = np.hstack([dataset, labels[:, np.newaxis]]).astype(np.float32)

        projects = np.split(dataset, project_splits)
        dataset = np.vstack([np.roll(p, 1, axis=0)[1:] for p in projects])

        split_mask = np.ones(len(labels), np.bool)
        split_mask[0] = False
        split_mask[project_splits] = False
        labels = labels[split_mask]
        weather = weather[split_mask]

        return dataset, labels, weather, latest, latest_labels, latest_indexes

    @staticmethod
    def _scale(train_data, test_data):
        # Scale the data to an appropriate normalized scale [0, 1) suitable for
        # training in normally weighted neural networks.
        scaler = MinMaxScaler((0, 1), copy=True)
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)

        return train_data, test_data

    def _weight_classes(self, labels):
        # Provide rebalancing weights.
        train_labels = labels[self.TRAIN]
        counts = np.bincount(train_labels)
        ratios = (counts / float(len(train_labels))).astype(np.float32)
        logging.debug('Ratios: %r', ratios)

        weights = [np.choose(set_labels, ratios) for set_labels in labels]
        return weights, ratios

    def _assemble_sets(self, dataset, labels, weather):
        train_data, test_data, train_labels, test_labels, train_weather, test_weather = \
            train_test_split(dataset, labels, weather,
                             test_size=self.args.test_size,
                             stratify=labels)

        train_data, test_data = self._scale(train_data, test_data)

        weights, self.ratios = \
            self._weight_classes([train_labels, test_labels])

        train = (train_data, train_labels, weights[self.TRAIN], train_weather)
        test = (test_data, test_labels, weights[self.TEST], test_weather)

        return train, test

    def load_datasets(self):
        """
        Load the dataset and split into train/test, and inputs/labels.
        """

        dataset, labels, project_splits, full_data = \
            self._select_data(*self._load())

        weather = self._last_sprint_weather_accuracy(labels, project_splits,
                                                     name='full dataset')[0]

        if self.args.roll_sprints:
            dataset, labels, weather, validation_data, validation_labels, validation_indexes = \
                self._roll_sprints(project_splits, dataset, labels, weather)
        else:
            validation_data = np.empty(0)
            validation_labels = np.empty(0)

        train, test = self._assemble_sets(dataset, labels, weather)

        self._last_sprint_weather_accuracy(train[self.LABELS],
                                           weather=train[self.WEATHER],
                                           name='training set')
        self._last_sprint_weather_accuracy(test[self.LABELS],
                                           weather=test[self.WEATHER],
                                           name='test set')

        validation_weights = np.full(validation_labels.shape, 0.5)
        validation = (validation_data, validation_labels, validation_weights, 0)
        self.data_sets = {
            self.TRAIN: train,
            self.TEST: test,
            self.VALIDATION: validation
        }
        self.validation_context = full_data[validation_indexes, :]
        self.num_features = train[self.INPUTS].shape[1]
        self.num_labels = max(labels) + 1

    def get_batches(self, data_set):
        """
        Create batches of the input/labels. This must be called at least once
        before a graph is set up, otherwise the batching will stall.
        """

        if data_set in self._batches:
            return self._batches[data_set]
        if data_set not in self.data_sets:
            raise IndexError('Data set #{} does not exist'.format(data_set))

        with tf.name_scope('input'):
            # Input data, pin to CPU because rest of pipeline is CPU-only
            with tf.device('/cpu:0'):
                inputs, labels, weights = [
                    tf.constant(item) for item in self.data_sets[data_set][0:3]
                ]

                # Only loop through the ordered validation set once
                if data_set == self.VALIDATION:
                    num_epochs = 1
                    shuffle = False
                else:
                    num_epochs = self.args.num_epochs
                    shuffle = True

                inputs, labels, weights = \
                    tf.train.slice_input_producer([inputs, labels, weights],
                                                  num_epochs=num_epochs,
                                                  shuffle=shuffle)

                if self.args.stratified_sample:
                    target_prob = [
                        1/float(self.num_labels) for _ in range(self.num_labels)
                    ]
                    kwargs = {
                        'init_probs': self.ratios,
                        'threads_per_queue': self.args.num_threads
                    }
                    tensors, labels = \
                        tf.contrib.training.stratified_sample([inputs, weights],
                                                              labels,
                                                              target_prob,
                                                              self.args.batch_size,
                                                              **kwargs)
                    inputs, weights = tensors[0]
                else:
                    inputs, labels, weights = \
                        tf.train.batch([inputs, labels, weights],
                                       batch_size=self.args.batch_size,
                                       num_threads=self.args.num_threads,
                                       allow_smaller_final_batch=True)

        self._batches[data_set] = [inputs, labels, weights]
        return self._batches[data_set]

    def clear_batches(self, data_set):
        """
        Remove cached batches.
        """

        if data_set in self._batches:
            del self._batches[data_set]
