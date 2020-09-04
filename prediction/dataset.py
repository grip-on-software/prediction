"""
TensorFlow ARFF dataset loader.
"""

from collections import OrderedDict
import itertools
import logging
import tensorflow as tf
from scipy.io import arff
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sortedcontainers import SortedSet
import expression
from .files import get_file_opener

class Loader(object):
    """
    Dataset loader.
    """

    FUNCTIONS = {
        'round': np.round,
        'mean': lambda *a: np.mean(a, axis=0),
        'where': np.where
    }

    def __init__(self, args):
        self.args = args

        self._full_data, self._meta = self._load()

        self._feature_meta = OrderedDict.fromkeys(self.names)

        self._indexes, self._labels, self._label_indexes = \
            self._calculate_indexes()
        self._combinations = None

    def translate(self, indices, translation=None):
        """
        Look up the index numbers in the full data set given a list of indices.
        Each entry in the indices may itself be a comma-separated list of lookup
        values, which may be indexes themselves or a feature name.

        The `translation`, if given, must be a dictionary mapping feature names
        to column indexes.

        Returns a generator which yield indexes until all the entries of the
        indices have been translated. If an index is outside of the allowable
        range of column indexes, or it is not a column index, translatable
        feature name or a comma-separated entry of indexes and feature names,
        then a `ValueError` is raised.
        """

        if translation is None:
            translation = self.name_translation

        for index in indices:
            try:
                if 0 <= int(index) < self.num_columns:
                    yield int(index)
                else:
                    raise ValueError('Index out of range: {0}'.format(index))
            except ValueError:
                if index in translation:
                    yield translation[index]
                elif ',' in index:
                    for idx in self.translate(index.split(','), translation):
                        yield idx
                else:
                    raise ValueError('Index {0} could not be understood'.format(index))

    def _get_labels(self, columns):
        parser = expression.Expression_Parser(variables=columns,
                                              functions=self.FUNCTIONS)
        column = parser.parse(self.args.label)
        label_indexes = set()
        if isinstance(column, int):
            label_indexes.add(column)
            column = self._full_data[:, column]
        elif not isinstance(column, np.ndarray):
            raise TypeError("Invalid label {0}: {1!r}".format(self.args.label,
                                                              type(column)))

        label_indexes.update(self.translate(parser.used_variables))

        if self.args.replace_na is not False:
            column[~np.isfinite(column)] = self.args.replace_na

        labels = column.astype(int)

        if self.args.binary is not None:
            labels = (column >= self.args.binary).astype(int)
        elif np.any(column - labels) != 0:
            raise ValueError('Label {0} produced non-round numbers'.format(self.args.label))

        return labels, label_indexes

    def _get_assignments(self, indexes):
        meta = OrderedDict(self._feature_meta)
        num_columns = self.num_columns
        new_columns = []
        parser = expression.Expression_Parser(variables=self.name_columns,
                                              functions=self.FUNCTIONS,
                                              assignment=True)
        for assignment in self.args.assign:
            parser.parse(assignment)
            if not parser.modified_variables:
                raise NameError('Expression must produce assignment')

            values = parser.modified_variables.values()
            if not all(isinstance(value, np.ndarray) for value in values):
                raise TypeError("Invalid assignment {}".format(assignment))

            for name in parser.modified_variables.keys():
                meta[name] = {
                    "attributes": parser.used_variables,
                    "expression": assignment
                }

            new_columns.extend(values)
            indexes.update(range(num_columns, num_columns + len(values)))
            num_columns += len(values)

        full_data = np.hstack([self._full_data, np.array(new_columns).T])
        return full_data, meta, num_columns

    def _load(self):
        file_opener = get_file_opener(self.args)
        with file_opener(self.args.filename) as features_file:
            data, meta = arff.loadarff(features_file)

        logging.debug('Metadata:\n%r', meta)

        translation = []
        for column in meta:
            if meta[column][0] == 'nominal':
                nominals = [text.encode('utf-8') for text in meta[column][1]]
                translation.append(dict(zip(nominals, range(len(nominals)))))
            else:
                translation.append({})

        full_data = np.array([[translation[index].get(cell, cell)
                               for index, cell in enumerate(row)]
                              for row in data], dtype=np.float32)

        return full_data, meta

    def _select_combination(self, indexes):
        if self._combinations is None:
            self._combinations = itertools.combinations(indexes,
                                                        self.args.combinations)

        return next(self._combinations)

    def _calculate_indexes(self):
        name_translation = self.name_translation
        if self.args.index:
            indexes = SortedSet(self.translate(self.args.index,
                                               translation=name_translation))
        else:
            indexes = SortedSet(range(self.num_columns))

        if not self.args.project:
            indexes.discard(Dataset.PROJECT_KEY)
        if "organization" in name_translation:
            if name_translation["organization"] != self.num_columns - 1:
                raise ValueError("Last column must be organization if included")
            indexes.discard(name_translation["organization"])

        if self.args.remove:
            indexes -= set(self.translate(self.args.remove,
                                          translation=name_translation))

        label_indexes = set()
        if self.args.label:
            labels, label_indexes = self._get_labels(self.name_columns)
            indexes -= label_indexes

            logging.debug('Selected labels %r: %r', label_indexes, labels)
        else:
            logging.info('No labels selected, generating random 2-class labels')
            labels = np.random.randint(0, 1+1, size=(self._full_data.shape[0],))

        if self.args.time:
            indexes.add(next(self.translate((self.args.time,))))

        self._feature_meta = OrderedDict.fromkeys(
            name for index, name in enumerate(self._feature_meta.keys())
            if index in indexes
        )
        logging.debug('Leftover indices: %r', indexes)
        logging.debug('Leftover column names: %r', self._feature_meta.keys())

        if self.args.assign:
            self._full_data, self._feature_meta = \
                self._get_assignments(indexes)[0:2]

        return indexes, labels, label_indexes

    @property
    def full_data(self):
        """
        Retrieve the full unfiltered data set.
        """

        return self._full_data

    @property
    def indexes(self):
        """
        Retrieve the set of indexes of selected attributes, before selecting
        combinations of those attributes.
        """

        return self._indexes

    @property
    def meta(self):
        """
        Retrieve the ARFF MetaData object describing the columns of the
        unfiltered data set.
        """

        return self._meta

    @property
    def label_indexes(self):
        """
        Retrieve the set of indexes of attributes used to generate the label
        values.
        """

        return self._label_indexes

    @property
    def names(self):
        """
        Retrieve the attribute names of the unfiltered data set.
        """

        return self._meta.names()

    @property
    def features(self):
        """
        Retrieve the names of the attributes that were selected as features.
        """

        return list(self._feature_meta.keys())

    @property
    def assignments(self):
        """
        Retrieve a dictionary of attributes that were generated from other
        features.
        """

        return dict(
            (name, features) for name, features in self._feature_meta.items()
            if features is not None
        )

    @property
    def labels(self):
        """
        Retrieve the names of the attributes that were used to build labels.
        """

        return [
            name for index, name in enumerate(self._meta)
            if index in self._label_indexes
        ]

    @property
    def num_columns(self):
        """
        Retrieve the number of attributes in the unfiltered data set.
        """

        return self._full_data.shape[1]

    @property
    def name_columns(self):
        """
        Retrieve a dictionary of names and column arrays of the attributes
        in the unfiltered data set.
        """

        return dict(zip(self.names, self._full_data.T))

    @property
    def project_split_keys(self):
        """
        Retrieve the column indexes of the attributes that indicate which
        project a record belongs to. Returns the index numbers as a tuple.
        """

        if "organization" in self.names:
            return (Dataset.PROJECT_KEY, Dataset.ORGANIZATION_KEY)

        return (Dataset.PROJECT_KEY,)

    @property
    def name_translation(self):
        """
        Retrieve a dictionary of names and column indexes of the attributes
        in the unfiltered data set.
        """

        return dict(zip(self.names, range(self.num_columns)))

    @property
    def feature_translation(self):
        """
        Retrieve a dictionary of names and column indexes in the filtered
        data set where only the remaining feature names are provided.
        """

        return dict(zip(self._feature_meta.keys(),
                        range(len(self._feature_meta))))

    @property
    def project_splits(self):
        """
        Retrieve the indexes of the samples in the data set at which a new
        project is introduced. Assumes that the data set is ordered on
        project identifier.
        """

        projects = self._full_data[:, self.project_split_keys]
        splits = np.unique(np.argwhere(np.diff(projects, axis=0) != 0)[:, 0])
        return np.squeeze(splits + 1)

    def select_data(self):
        """
        Retrieve the dataset and labels based on selection criteria.
        """

        if self.args.combinations:
            indexes = self._select_combination(self._indexes)
        else:
            indexes = self._indexes

        logging.debug('Selected combination of indexes: %r', indexes)
        dataset = self._full_data[:, tuple(indexes)]
        if self.args.replace_na is not False:
            dataset[~np.isfinite(dataset)] = self.args.replace_na
        return dataset, indexes, self._labels

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
    INDEXES = 4

    # Indexes in the original dataset of uniquely identifying keys
    PROJECT_KEY = 0
    SPRINT_KEY = 1
    ORGANIZATION_KEY = -1

    def __init__(self, args):
        self.args = args

        self.data_sets = None
        self.num_labels = None

        self._loader = Loader(self.args)
        self._feature_indexes = None
        self._times = None
        self.load_datasets()

        self._batches = {}
        self.ratios = None

    @classmethod
    def _last_sprint_weather(cls, labels, project_splits):
        project_labels = np.split(labels, project_splits)
        return np.hstack([
            np.hstack([np.nan, np.roll(project_label, 1, axis=0)[1:]])
            for project_label in project_labels if project_label.size != 0
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

    def _roll(self, project_data):
        rolls = []

        for i in range(1, self.args.roll_sprints+1):
            rolled_data = np.roll(project_data, i, axis=0)

            # Mark earliest rolled values as missing
            rolled_data[:i, :] = np.nan

            rolls.append(rolled_data)

        if self.args.keep_index:
            indexes = self._loader.translate(self.args.keep_index,
                                             self._loader.feature_translation)
            rolls.append(project_data[:, list(indexes)])

        return np.hstack(rolls)

    def _get_splits(self, project_splits, dataset):
        # Get rolled splits
        return [
            self._roll(project) for project in np.split(dataset, project_splits)
            if project.size != 0
        ]

    def _trim(self, projects, end=True):
        # After rolling, the first sample is always empty, but we may wish to
        # keep samples with only a few sprints worth of features.
        trim_start = 1 if self.args.keep_incomplete else self.args.roll_sprints

        if end:
            trim_end = -2 if self.args.roll_validation else -1
            return np.vstack([p[trim_start:trim_end] for p in projects])

        return np.vstack([p[trim_start:] for p in projects])

    def _get_split_mask(self, project_splits, labels, validation_mask=None):
        split_mask = np.ones(len(labels), np.bool)

        # Remove labels and weather data from the normal train/test dataset for
        # removed (incomplete) samples and samples used for validation set
        if self.args.keep_incomplete:
            split_mask[0] = False
            split_mask[project_splits] = False
        else:
            split_mask[0:self.args.roll_sprints] = False

            project_trims = np.hstack([
                (project_splits+i)[project_splits+i < len(labels)]
                for i in range(self.args.roll_sprints)
            ])
            split_mask[project_trims] = False

        if validation_mask is not None:
            split_mask[validation_mask] = False
        else:
            split_mask[project_splits-1] = False
            split_mask[-1] = False
            if self.args.roll_validation:
                split_mask[project_splits-2] = False
                split_mask[-2] = False

        return split_mask

    def _roll_sprints(self, project_splits, dataset, labels, weather):
        # Roll the sprints such that a sprint has features from the sprint
        # before it, while the labels remain the same for that sprint.
        # Remove the sprint at the start of a project in lack of features.
        logging.debug('Project splits: %r', project_splits)

        if self.args.roll_labels:
            # Roll the labels of the previous sprint into the features of
            # the current sprint, like how last sprint's weather classification
            # has access to this label as well.
            dataset = np.hstack([dataset, labels[:, np.newaxis]]).astype(np.float32)

        # Keep track of original indexes in the dataset, features and labels
        indexes = np.arange(len(dataset))
        projects = self._get_splits(project_splits, dataset)

        if self.args.validation_index:
            i = next(self._loader.translate((self.args.validation_index,),
                                            self._loader.feature_translation))
            validation_mask = dataset[:, i] != 0
            if self.args.roll_validation:
                validation_mask[project_splits-2] = True
            else:
                validation_mask[project_splits-1] = True

            latest_indexes = indexes[validation_mask]

            latest_mask = self._trim(np.split(np.reshape(validation_mask,
                                                         (dataset.shape[0], 1)),
                                              project_splits),
                                     end=False).flatten()
            dataset = self._trim(projects, end=False)

            latest_data = dataset[latest_mask, :]
            dataset = dataset[~latest_mask, :]
        else:
            latest_indexes = np.hstack([project_splits-1, -1])
            # Obtain the correct validation rows based on whether validation
            # features are rolled.
            # Do not alter the indexes because the context from the full data
            # remains the same. Labels are also not rolled.
            if self.args.roll_validation:
                latest_data = np.vstack([p[-1, :] for p in projects])
            else:
                latest_data = np.vstack([p[-2:, :][0, :] for p in projects])

            dataset = self._trim(projects)
            validation_mask = None

        latest_labels = labels[latest_indexes]

        split_mask = self._get_split_mask(project_splits, labels,
                                          validation_mask=validation_mask)

        labels = labels[split_mask]
        weather = weather[split_mask]
        indexes = indexes[split_mask]
        latest = (latest_data, latest_labels,
                  np.full(latest_labels.shape, 0.5), None, latest_indexes)

        return dataset, labels, weather, latest, indexes

    def _scale(self, datasets):
        # Scale the data to an appropriate normalized scale [0, 1) suitable for
        # training in normally weighted neural networks.
        scaler = MinMaxScaler((0, 1), copy=True)
        train_data = datasets[self.TRAIN]
        scaler.fit(train_data[np.isfinite(train_data).all(axis=1)])

        return [dataset * scaler.scale_ + scaler.min_ for dataset in datasets]

    @staticmethod
    def choose(data, choices):
        """
        Variant of numpy.choose which does not limit the use of arrays with
        a large dimension.
        """

        options = np.broadcast_to(choices, [data.shape[-1], choices.shape[-1]])
        return options[range(len(data)), data]

    def _weight_classes(self, labels):
        # Provide rebalancing weights.
        train_labels = labels[self.TRAIN]
        counts = np.bincount(train_labels, minlength=self.num_labels)
        ratios = (counts / float(len(train_labels))).astype(np.float32)
        logging.debug('Ratios: %r', ratios)

        weights = [self.choose(set_labels, ratios) for set_labels in labels]
        return weights, ratios

    def _assemble_time_sets(self, dataset, labels, weather, time_index, current_time):
        indexes = np.arange(len(dataset))

        features = [True] * dataset.shape[1]
        logging.info('%r', len(self._loader.features))
        for i in range(0, self.args.roll_sprints+1):
            offset = i * (len(self._loader.features) + len(self.args.keep_index))
            logging.info('%d %d', offset, time_index)
            features[offset + time_index] = False

        logging.info('%r', features)

        train_data = dataset[dataset[:, time_index] < current_time, :][:, features]
        test_data = dataset[dataset[:, time_index] > current_time, :][:, features]
        validation_data = dataset[dataset[:, time_index] == current_time, :][:, features]

        train_labels = labels[dataset[:, time_index] < current_time]
        test_labels = labels[dataset[:, time_index] > current_time]
        validation_labels = labels[dataset[:, time_index] == current_time]

        train_weather = weather[dataset[:, time_index] < current_time]
        test_weather = weather[dataset[:, time_index] > current_time]

        train_indexes = indexes[dataset[:, time_index] < current_time]
        test_indexes = indexes[dataset[:, time_index] > current_time]
        validation_indexes = indexes[dataset[:, time_index] == current_time]

        train_data, test_data, validation_data = \
            self._scale([train_data, test_data, validation_data])

        weights, self.ratios = \
            self._weight_classes([train_labels, test_labels])

        return [
            (train_data, train_labels, weights[self.TRAIN], train_weather,
             train_indexes),
            (test_data, test_labels, weights[self.TEST], test_weather,
             test_indexes),
            (validation_data, validation_labels,
             np.full(validation_labels.shape, 0.5), 0.0, validation_indexes)
        ]

    def _assemble_sets(self, validation, *items):
        train_data, test_data, \
            train_labels, test_labels, \
            train_weather, test_weather, \
            train_indexes, test_indexes = \
            train_test_split(*items,
                             test_size=self.args.test_size,
                             stratify=items[self.LABELS] if self.args.stratified_split else None)

        train_data, test_data, validation_data = \
            self._scale([train_data, test_data, validation[self.INPUTS]])

        weights, self.ratios = \
            self._weight_classes([train_labels, test_labels])

        return [
            (train_data, train_labels, weights[self.TRAIN], train_weather,
             train_indexes),
            (test_data, test_labels, weights[self.TEST], test_weather,
             test_indexes),
            (validation_data, validation[self.LABELS],
             validation[self.WEIGHTS], 0.0, validation[self.INDEXES])
        ]

    def load_datasets(self):
        """
        Load the dataset and split into train/test, and inputs/labels.
        """

        dataset, self._feature_indexes, labels = self._loader.select_data()
        project_splits = self._loader.project_splits

        weather = self._last_sprint_weather_accuracy(labels, project_splits,
                                                     name='full dataset')[0]

        if self.args.roll_sprints > 0:
            dataset, labels, weather, validation, indexes = \
                self._roll_sprints(project_splits, dataset, labels, weather)
        else:
            logging.info('Cannot generate a validation set by rolling sprints')
            validation = (np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0))
            indexes = np.empty(0)

        self.num_labels = max(labels) + 1

        if self.args.time:
            time_index = \
                next(self._loader.translate((self.args.time,),
                                            self._loader.feature_translation))
            logging.info('Time index: %r', time_index)

            logging.info('%r', dataset[0,:])
            if self._times is None:
                self._times = iter(set(dataset[:, time_index]))
                # Need at least one train instance.
                for _ in range(self.args.time_skip):
                    next(self._times)
            current_time = next(self._times)
            logging.warning('Current time is %s', current_time)
            train, test, validation = \
                self._assemble_time_sets(dataset, labels, weather, time_index, current_time)
        else:
            train, test, validation = \
                self._assemble_sets(validation, dataset, labels, weather, indexes)

        self._last_sprint_weather_accuracy(train[self.LABELS],
                                           weather=train[self.WEATHER],
                                           name='training set')
        self._last_sprint_weather_accuracy(test[self.LABELS],
                                           weather=test[self.WEATHER],
                                           name='test set')

        self.data_sets = {
            self.TRAIN: train,
            self.TEST: test,
            self.VALIDATION: validation
        }

        # Show properties for validation sprints that uniquely identify them.
        validation_context = self.get_context(self.VALIDATION)
        keys = self._loader.project_split_keys + (self.SPRINT_KEY,)
        logging.info('Validation sprints: %r', validation_context[:, keys])

    @property
    def organizations(self):
        """
        Retrieve the list of organization names, ordered by the indexes as used
        in the numeric data set. If no organizations were included in the data
        set, then an empty list is returned.
        """

        if "organization" in self._loader.names:
            return self._loader.meta["organization"][1]

        return []

    @property
    def features(self):
        """
        Retrieve the names of the features used in the data set.
        """

        return self._loader.features

    @property
    def assignments(self):
        """
        Retrieve the assignment variables for generated features as a dictionary
        of feature name and set of attribute names.
        """

        return self._loader.assignments

    @property
    def labels(self):
        """
        Retrieve the names of the labels used in the data set.
        """

        return self._loader.labels

    @property
    def num_features(self):
        """
        Retrieve the number of features in the data set.
        """

        return self.data_sets[self.TRAIN][self.INPUTS].shape[1]

    def get_context(self, dataset):
        """
        Retrieve the original samples from the data set from which the
        given data set is derived. This contains all feature/label columns
        as well as contextual information that is discarded from the data
        set such as project identifiers.
        """

        indexes = self.data_sets[dataset][self.INDEXES]
        return self._loader.full_data[indexes, :]

    def get_values(self, dataset):
        """
        Retrieve the values of the features from the given data set.
        This only contains the features that are provided to the model.
        """

        context = self.get_context(dataset)
        return context[:, list(self._feature_indexes)]

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
            # Input data
            with tf.device(self.args.device):
                inputs, labels, weights = [
                    tf.constant(item) for item in self.data_sets[data_set][0:3]
                ]
                # Indexes within the set
                indexes = tf.constant(list(range(len(self.data_sets[data_set][0]))))
                weather = tf.constant(self.data_sets[data_set][self.WEATHER])

                # Only loop through the validation set once and remain order.
                if data_set == self.VALIDATION:
                    num_epochs = 1
                    capacity = len(self.data_sets[data_set][0])
                    shuffle = False
                else:
                    num_epochs = self.args.num_epochs
                    capacity = 32
                    shuffle = True

                inputs, labels, weights, indexes = \
                    tf.train.slice_input_producer([inputs, labels, weights, indexes],
                                                  num_epochs=num_epochs,
                                                  capacity=capacity,
                                                  shuffle=shuffle,
                                                  seed=self.args.seed)

                if self.args.stratified_sample:
                    target_prob = [
                        1/float(self.num_labels) for _ in range(self.num_labels)
                    ]
                    kwargs = {
                        'queue_capacity': capacity,
                        'init_probs': self.ratios,
                        'threads_per_queue': self.args.num_threads
                    }
                    tensors, labels = \
                        tf.contrib.training.stratified_sample([inputs, weights, indexes],
                                                              labels,
                                                              target_prob,
                                                              self.args.batch_size,
                                                              **kwargs)
                    inputs, weights, indexes = tensors
                else:
                    inputs, labels, weights, indexes = \
                        tf.train.batch([inputs, labels, weights, indexes],
                                       batch_size=self.args.batch_size,
                                       capacity=capacity,
                                       num_threads=self.args.num_threads,
                                       allow_smaller_final_batch=True)

        self._batches[data_set] = [inputs, labels, weights, weather, indexes]
        return self._batches[data_set]

    def clear_batches(self, data_set):
        """
        Remove cached batches.
        """

        if data_set in self._batches:
            del self._batches[data_set]
