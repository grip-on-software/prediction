"""
TensorFlow ARFF dataset loader.
"""

import itertools
import logging
import tensorflow as tf
from scipy.io import arff
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import expression
from .files import get_file_opener

class Loader(object):
    """
    Dataset loader.
    """

    def __init__(self, args):
        self.args = args

        self._full_data, self._meta = self._load()
        self._indexes, self._labels = self._calculate_indexes()
        self._selected_data = None
        self._combinations = None

    def _translate(self, indices, translation):
        for index in indices:
            try:
                yield int(index)
            except ValueError:
                if index in translation:
                    yield translation[index]
                elif ',' in index:
                    for idx in self._translate(index.split(','), translation):
                        yield idx
                else:
                    raise ValueError('Index {0} could not be understood'.format(index))

    def _get_labels(self, columns, translation):
        parser = expression.Expression_Parser(variables=columns,
                                              functions={'round': np.round})
        column = parser.parse(self.args.label)
        label_indexes = set()
        if isinstance(column, int):
            label_indexes.add(column)
            column = self._full_data[:, column]
        elif not isinstance(column, np.ndarray):
            raise TypeError("Invalid label {0}: {1!r}".format(self.args.label,
                                                              type(column)))

        label_indexes.update(self._translate(parser.used_variables, translation))

        if self.args.replace_na is not False:
            column[~np.isfinite(column)] = self.args.replace_na

        labels = column.astype(int)

        if self.args.binary is not None:
            labels = (column >= self.args.binary).astype(int)
        elif np.any(column - labels) != 0:
            raise ValueError('Label {0} produced non-round numbers'.format(self.args.label))

        return labels, label_indexes

    def _get_assignments(self, indexes):
        num_columns = self.num_columns
        new_columns = []
        parser = expression.Expression_Parser(variables=self.name_columns,
                                              assignment=True)
        for assignment in self.args.assign:
            parser.parse(assignment)
            if not parser.modified_variables:
                raise NameError('Expression must produce assignment')

            values = parser.modified_variables.values()
            if not all(isinstance(value, np.ndarray) for value in values):
                raise TypeError("Invalid assignment {}".format(assignment))

            new_columns.extend(values)
            indexes.update(range(num_columns, num_columns + len(values)))
            num_columns += len(values)

        full_data = np.hstack([self._full_data, np.array(new_columns).T])
        return full_data, num_columns

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
        name_translation = dict(zip(self.names, range(self.num_columns)))

        if self.args.index:
            indexes = set(self._translate(self.args.index, name_translation))
        else:
            indexes = set(range(self.num_columns))

        indexes.discard(Dataset.PROJECT_KEY)
        if "organization" in name_translation:
            if name_translation["organization"] != self.num_columns - 1:
                raise ValueError("Last column must be organization if included")
            indexes.discard(name_translation["organization"])

        if self.args.remove:
            indexes -= set(self._translate(self.args.remove, name_translation))

        if self.args.label:
            labels, label_indexes = self._get_labels(self.name_columns,
                                                     name_translation)
            indexes -= label_indexes

            logging.debug('Selected labels %r: %r', label_indexes, labels)
        else:
            logging.info('No labels selected, generating random 2-class labels')
            labels = np.random.randint(0, 1+1, size=(self._full_data.shape[0],))

        names = [name for index, name in enumerate(self._meta) if index in indexes]
        logging.debug('Leftover indices: %r', indexes)
        logging.debug('Leftover column names: %r', names)

        if self.args.assign:
            self._full_data = self._get_assignments(indexes)[0]

        return indexes, labels

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
    def names(self):
        """
        Retrieve the attribute names.
        """

        return self._meta.names()

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
        return dataset, self._labels

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
    INDEXES = 3

    # Indexes in the original dataset of uniquely identifying keys
    PROJECT_KEY = 0
    SPRINT_KEY = 1
    ORGANIZATION_KEY = -1

    def __init__(self, args):
        self.args = args

        self.data_sets = None
        self.num_labels = None

        self._loader = Loader(self.args)
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

        return np.hstack(rolls)

    def _get_splits(self, project_splits, dataset):
        # Get rolled splits
        return [
            self._roll(project) for project in np.split(dataset, project_splits)
            if project.size != 0
        ]

    def _trim(self, projects):
        # After rolling, the first sample is always empty, but we may wish to
        # keep samples with only a few sprints worth of features.
        trim_start = 1 if self.args.keep_incomplete else self.args.roll_sprints
        trim_end = -2 if self.args.roll_validation else -1
        return np.vstack([p[trim_start:trim_end] for p in projects])

    def _get_split_mask(self, project_splits, labels):
        split_mask = np.ones(len(labels), np.bool)

        # Remove labels and weather data from the normal train/test dataset for
        # removed (incomplete) samples and samples used for validation set
        split_mask[project_splits-1] = False
        split_mask[-1] = False
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

        # Validation data: original indexes in the dataset, features and labels
        latest_indexes = np.hstack([project_splits-1, -1])
        projects = self._get_splits(project_splits, dataset)

        # Obtain the correct validation rows based on whether validation
        # features are rolled.
        # Do not alter the indexes because the context from the full data
        # remains the same. Labels are also not rolled.
        if self.args.roll_validation:
            latest_data = np.vstack([p[-1, :] for p in projects])
        else:
            latest_data = np.vstack([p[-2, :] for p in projects])

        latest_labels = labels[latest_indexes]

        dataset = self._trim(projects)
        split_mask = self._get_split_mask(project_splits, labels)

        logging.debug('%r', split_mask)

        labels = labels[split_mask]
        weather = weather[split_mask]
        latest_weights = np.full(latest_labels.shape, 0.5)
        latest = (latest_data, latest_labels, latest_weights, latest_indexes)

        return dataset, labels, weather, latest

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

    def _assemble_sets(self, dataset, labels, weather, validation):
        train_data, test_data, train_labels, test_labels, train_weather, test_weather = \
            train_test_split(dataset, labels, weather,
                             test_size=self.args.test_size,
                             stratify=labels if self.args.stratified_split else None)

        train_data, test_data, validation_data = \
            self._scale([train_data, test_data, validation[self.INPUTS]])

        weights, self.ratios = \
            self._weight_classes([train_labels, test_labels])

        return [
            (train_data, train_labels, weights[self.TRAIN], train_weather),
            (test_data, test_labels, weights[self.TEST], test_weather),
            (validation_data, validation[self.LABELS],
             validation[self.WEIGHTS], validation[self.INDEXES])
        ]

    def load_datasets(self):
        """
        Load the dataset and split into train/test, and inputs/labels.
        """

        dataset, labels = self._loader.select_data()
        project_splits = self._loader.project_splits

        weather = self._last_sprint_weather_accuracy(labels, project_splits,
                                                     name='full dataset')[0]

        if self.args.roll_sprints > 0:
            dataset, labels, weather, validation = \
                self._roll_sprints(project_splits, dataset, labels, weather)
        else:
            logging.info('Cannot generate a validation set by rolling sprints')
            validation = (np.empty(0), np.empty(0), np.empty(0), np.empty(0))

        self.num_labels = max(labels) + 1

        train, test, validation = \
            self._assemble_sets(dataset, labels, weather, validation)

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
        keys = self._loader.project_split_keys + (self.SPRINT_KEY,)
        logging.info('Validation sprints: %r', self.validation_context[:, keys])

    @property
    def num_features(self):
        """
        Retrieve the number of features in the data set.
        """

        return self.data_sets[self.TRAIN][self.INPUTS].shape[1]

    @property
    def validation_context(self):
        """
        Retrieve the original samples from the data set from which the
        validation set is derived. This contains all feature/label columns.
        """

        validation_indexes = self.data_sets[self.VALIDATION][self.INDEXES]
        return self._loader.full_data[validation_indexes, :]

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
                indexes = tf.constant(list(range(len(self.data_sets[data_set][0]))))

                # Only loop through the validation set once and remain order.
                if data_set == self.VALIDATION:
                    num_epochs = 1
                    shuffle = False
                else:
                    num_epochs = self.args.num_epochs
                    shuffle = True

                inputs, labels, weights, indexes = \
                    tf.train.slice_input_producer([inputs, labels, weights, indexes],
                                                  num_epochs=num_epochs,
                                                  shuffle=shuffle,
                                                  seed=self.args.seed)

                if self.args.stratified_sample:
                    target_prob = [
                        1/float(self.num_labels) for _ in range(self.num_labels)
                    ]
                    kwargs = {
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
                                       num_threads=self.args.num_threads,
                                       allow_smaller_final_batch=True)

        self._batches[data_set] = [inputs, labels, weights, indexes]
        return self._batches[data_set]

    def clear_batches(self, data_set):
        """
        Remove cached batches.
        """

        if data_set in self._batches:
            del self._batches[data_set]
