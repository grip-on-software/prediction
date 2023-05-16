"""
TensorFlow prediction models.

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

import logging
import math
import tensorflow as tf
import keras.models as km
import keras.layers as kl
try:
    from dbn.tensorflow import SupervisedDBNClassification
except ImportError as error:
    SupervisedDBNClassification = None
from .dataset import Dataset
from .runner import TFRunner, TFEstimatorRunner, TFSKLRunner, KerasRunner, FullTrainRunner

class Model:
    """
    A generic prediction/classification model that can be trained/optimized
    through TensorFlow graphs.
    """

    # Index of the loss op.
    LOSS_OP = 1

    RUNNER = TFRunner

    _models = {}

    @classmethod
    def register(cls, name):
        """
        Register a model by its short name.
        """

        def decorator(target):
            """
            Decorator for Model classes.
            """

            cls._models[name] = target
            return target

        return decorator

    @classmethod
    def get_model_names(cls):
        """
        Retrieve a list of short names for the registered models.
        """

        return list(cls._models.keys())

    @classmethod
    def get_model(cls, name):
        """
        Retrieve the Model class associated with the given `name`.
        """

        return cls._models[name]

    @classmethod
    def add_arguments(cls, parser):
        """
        Add arguments to an argument parser.
        """

    def __init__(self, args, dtypes, sizes):
        self.args = args
        self._num_features, self._num_labels = sizes
        input_dtype, label_dtype, weight_dtype = dtypes
        self._placeholders = {
            'x_input': tf.placeholder(dtype=input_dtype,
                                      shape=[None, self._num_features]),
            'y_labels': tf.placeholder(dtype=label_dtype, shape=[None]),
            'y_weights': tf.placeholder(dtype=weight_dtype, shape=[None])
        }

        self._outputs = None
        self._train_ops = []

    @property
    def num_features(self):
        """
        Retrieve the number of input features that the model accepts.
        """

        return self._num_features

    @property
    def num_labels(self):
        """
        Retrieve the number of output labels that the model can provide.
        """

        return self._num_labels

    @property
    def x_input(self):
        """
        Retrieve the input placeholder variable.
        """

        return self._placeholders['x_input']

    @property
    def y_labels(self):
        """
        Retrieve the output placeholder variable.
        """

        return self._placeholders['y_labels']

    @property
    def y_weights(self):
        """
        Retrieve the class weight ratio placeholder variable.
        """

        return self._placeholders['y_weights']

    @property
    def outputs(self):
        """
        Retrieve the actual outputs of the model.
        """

        return self._outputs

    @property
    def train_ops(self):
        """
        Retrieve the ops to run when training the model.
        """

        return self._train_ops

    def build(self):
        """
        Build the model.
        """

        raise NotImplementedError('Must be implemented by subclasses')

    def log_evaluation(self, session, feed_dict):
        """
        Provide logging output describing model internals after performing
        a test or validation operation to show the model's state. Member
        variables relevated to that op may be logged, but no new operations
        are to be run.
        """

    @property
    def validation_results(self):
        """
        Provide additional tensors containing results that the model has in
        addition to the data that the runner obtains through prediction after
        a validation operation. Member variables relevated to that op may be
        provided in a dictionary.
        """

        return {}

    @property
    def validation_metadata(self):
        """
        Provide a dictionary of the same shape as `validation_results`
        which describes what kind of results are returned.
        """

        return {}

@Model.register('mlp')
class MultiLayerPerceptron(Model):
    """
    Neural network model with multiple (visible, hidden, ..., output) layers.
    """

    RUNNER = TFRunner

    def __init__(self, args, dtypes, sizes):
        super().__init__(args, dtypes, sizes)
        self.weights1 = None
        self.biases1 = None
        self.weights2 = None
        self.biases2 = None
        self.weights_max = None
        self.biases_max = None

    @classmethod
    def add_arguments(cls, parser):
        group = parser.add_argument_group('MLP', 'Multi-layered perceptron')
        group.add_argument('--num-hidden1', dest='num_hidden1', type=int,
                           default=128, help='Number of units in hidden layer 1')
        group.add_argument('--num-hidden2', dest='num_hidden2', type=int,
                           default=32, help='Number of units in hidden layer 2')
        group.add_argument('--activation',
                           choices=['softmax', 'sigmoid', 'sigmax', 'onehot'],
                           default='softmax', help='Activation of final layer')

    @staticmethod
    def make_layer(name, inputs, num_visible, num_hidden, activation=True):
        """
        Make a layer with weights and biases based on the input shapes.
        """

        with tf.name_scope(name):
            stddev = 1.0 / math.sqrt(float(num_visible))
            weights = tf.Variable(tf.truncated_normal([num_visible, num_hidden],
                                                      stddev=stddev),
                                  name='weights')
            biases = tf.Variable(tf.random_normal([num_hidden]), name='biases')
            layer = tf.add(tf.matmul(inputs, weights), biases)
            if activation:
                layer = tf.nn.relu(layer)

        return layer, weights, biases

    def build(self):
        hidden1, self.weights1, self.biases1 = \
            self.make_layer('hidden1', self.x_input, self.num_features,
                            self.args.num_hidden1)
        hidden2, self.weights2, self.biases2 = \
            self.make_layer('hidden2', hidden1, self.args.num_hidden1,
                            self.args.num_hidden2)
        outputs, self.weights_max, self.biases_max = \
            self.make_layer('softmax_linear', hidden2, self.args.num_hidden2,
                            self.num_labels, activation=False)

        entropy_funcs = {
            'softmax': tf.losses.sparse_softmax_cross_entropy,
            'onehot': tf.losses.sparse_softmax_cross_entropy,
            'sigmoid': tf.losses.sigmoid_cross_entropy,
            'sigmax': tf.losses.sigmoid_cross_entropy
        }
        entropy_func = entropy_funcs[self.args.activation]

        onehot = tf.one_hot(self.y_labels, self.num_labels)
        if self.args.activation in ('sigmoid', 'sigmax'):
            labels = onehot
            weights = tf.one_hot(self.y_labels, self.num_labels,
                                 on_value=tf.reduce_max(self.y_weights),
                                 off_value=tf.reduce_min(self.y_weights))
        else:
            labels = self.y_labels
            weights = self.y_weights

        if not self.args.weighted:
            weights = 1

        loss = tf.reduce_mean(entropy_func(labels, outputs, weights=weights),
                              name='xentropy_mean')

        if self.args.activation == 'onehot':
            self._outputs = onehot
        elif self.args.activation == 'sigmoid':
            self._outputs = tf.sigmoid(outputs)
        elif self.args.activation == 'sigmax':
            self._outputs = tf.cast(tf.greater(tf.sigmoid(outputs), 0.5),
                                    'float')
        else:
            self._outputs = outputs

        self._train_ops.append(self.make_training(loss))
        self._train_ops.append(loss)

    def make_training(self, loss):
        """
        Set up the training operation.
        """

        tf.summary.scalar('loss', loss)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(self.args.learning_rate)
        #optimizer = tf.train.AdamOptimizer(self.args.learning_rate)
        return optimizer.minimize(loss, global_step=global_step)

    def log_evaluation(self, session, feed_dict):
        logging.debug("w1: %r b1: %r",
                      session.run(self.weights1, feed_dict=feed_dict),
                      session.run(self.biases1, feed_dict=feed_dict))
        logging.debug("w2: %r b2: %r",
                      session.run(self.weights2, feed_dict=feed_dict),
                      session.run(self.biases2, feed_dict=feed_dict))
        logging.debug("wm: %r bm: %r",
                      session.run(self.weights_max, feed_dict=feed_dict),
                      session.run(self.biases_max, feed_dict=feed_dict))

def euclidean_distance(_, batch_input, train_inputs):
    """
    Calculate euclidean distance for nearby inputs.
    """

    delta = batch_input[:, tf.newaxis, :] - train_inputs[tf.newaxis, :, :]
    return tf.reduce_sum(delta ** 2, axis=-1) ** 0.5

def cosine_distance(_, batch_input, train_inputs):
    """
    Calculate cosine distance for nearby inputs.
    """

    left = tf.nn.l2_normalize(batch_input, 1)
    right = tf.nn.l2_normalize(train_inputs, 1)
    return tf.matmul(left, right, adjoint_b=True)

def minkowsky_distance(args, batch_input, train_inputs):
    """
    Calculate Minkowsky distance for nearby inputs.
    """

    exp = tf.constant(args.exponent, dtype=tf.float32)
    left = tf.matmul(tf.expand_dims(tf.reduce_sum(tf.pow(batch_input, exp),
                                                  1), 1),
                     tf.ones(shape=[1, tf.shape(train_inputs)[0]]))

    right = tf.matmul(tf.reshape(tf.reduce_sum(tf.pow(train_inputs, exp),
                                               1), shape=[-1, 1]),
                      tf.ones(shape=[tf.shape(batch_input)[0], 1]),
                      transpose_b=True)

    return tf.subtract(tf.pow(tf.add(left, tf.transpose(right)),
                              tf.reciprocal(exp)),
                       2 * tf.matmul(batch_input, train_inputs,
                                     transpose_b=True))

@Model.register('abe')
class AnalogyBasedEstimation(Model):
    # pylint: disable=too-many-instance-attributes
    """
    Model that uses analogy-based estimation to predict an output
    """

    RUNNER = FullTrainRunner
    _distances = {
        'euclidean': euclidean_distance,
        'cosine': cosine_distance,
        'minkowsky': minkowsky_distance
    }

    def __init__(self, args, dtypes, sizes):
        super().__init__(args, dtypes, sizes)

        # Placeholders for the full training set
        self.train_inputs = tf.placeholder(dtype=dtypes[0],
                                           shape=[None, self._num_features])
        self.train_labels = tf.placeholder(dtype=dtypes[1], shape=[None])
        self.features = tf.get_variable("abe_features",
                                        shape=[self._num_features],
                                        initializer=tf.ones_initializer,
                                        trainable=False)

        self.values = None
        self.indices = None
        self.labels = None
        self._pred = None

    @classmethod
    def add_arguments(cls, parser):
        group = parser.add_argument_group('ABE', 'Analogy-based estimation')
        group.add_argument('--num-k', dest='num_k', type=int,
                           default=3, help='Number of neighbors to include')
        group.add_argument('--pred', type=float, default=0.25,
                           help='Percentage of tolerance from actual value')
        group.add_argument('--distance', choices=cls._distances.keys(),
                           default='minkowsky', help='Distance measure to use')
        group.add_argument('--exponent', type=float, default=2.0,
                           help='Exponent of Minkowsky L-P distance measure')

    def build(self):
        weighted_inputs = tf.multiply(self.features, self.train_inputs)

        measure = self._distances[self.args.distance]
        distance = measure(self.args, self.x_input, weighted_inputs)

        # Take values and indices of lowest distances
        self.values, self.indices = tf.nn.top_k(tf.negative(distance),
                                                k=self.args.num_k)
        label_shape = [tf.shape(self.x_input)[0], self.args.num_k]
        self.labels = tf.reshape(tf.gather(self.train_labels, self.indices),
                                 label_shape)
        outputs = tf.reduce_mean(self.labels, axis=1)

        # No specialized train op yet; select indices within op and use MMRE
        self._outputs = tf.one_hot(outputs, self.num_labels)
        magnitude = tf.divide(tf.abs(tf.subtract(self.y_labels, outputs)),
                              tf.add(self.y_labels, 1))

        # Calculate the Pred(x) metric for evaluation logging.
        self._pred = tf.divide(tf.reduce_sum(tf.cast(tf.less(magnitude,
                                                             self.args.pred),
                                                     tf.int32)),
                               tf.shape(self.y_labels)[0])

        self._train_ops.append(distance)
        self._train_ops.append(magnitude)

    def log_evaluation(self, session, feed_dict):
        logging.debug('Values: %r Indices: %r',
                      session.run(self.values, feed_dict=feed_dict),
                      session.run(self.indices, feed_dict=feed_dict))
        logging.debug('Labels: %r',
                      session.run(self.labels, feed_dict=feed_dict))
        logging.warning('Pred(%f): %f', self.args.pred,
                        session.run(self._pred, feed_dict=feed_dict))

    @property
    def validation_results(self):
        return {
            "analogy_distances": self.values,
            "analogy_indexes": self.indices,
            "analogy_labels": self.labels,
            "analogy_values": self.indices
        }

    @property
    def validation_metadata(self):
        return {
            "analogy_indexes": {
                "context": Dataset.TRAIN,
                "item": Dataset.INDEXES
            },
            "analogy_values": {
                "context": Dataset.TRAIN,
                "values": True
            }
        }

class EstimatorModel(Model):
    """
    Model based on a TensorFlow estimator.
    """

    RUNNER = TFEstimatorRunner

    INPUT_COLUMN = "x"
    WEIGHT_COLUMN = "weight"

    def __init__(self, args, dtypes, sizes):
        super().__init__(args, dtypes, sizes)
        self.predictor = None
        self.columns = [
            tf.contrib.layers.real_valued_column(self.INPUT_COLUMN,
                                                 dimension=self.num_features,
                                                 dtype=self.x_input.dtype)
        ]
        if self.args.weighted:
            self.columns.extend([
                tf.contrib.layers.real_valued_column(self.WEIGHT_COLUMN,
                                                     dtype=self.y_weights.dtype)
            ])

    def build(self):
        raise NotImplementedError('Must be implemented by subclasses')

@Model.register('dnn')
class DNNModel(EstimatorModel):
    """
    Deep neural network model from tf.estimator.
    """

    @classmethod
    def add_arguments(cls, parser):
        group = parser.add_argument_group('DNN', 'Deep neural network')
        group.add_argument('--hiddens', nargs='+', type=int,
                           default=[100, 150, 100],
                           help='Number of units per hidden layer')

    def build(self):
        # pylint: disable=no-member
        train_interval = self.args.train_interval
        num_checkpoints = self.args.num_checkpoints if self.args.save else 0
        run_config = \
            tf.estimator.RunConfig(log_step_count_steps=train_interval,
                                   save_checkpoints_steps=train_interval,
                                   keep_checkpoint_max=num_checkpoints,
                                   save_summary_steps=train_interval,
                                   model_dir=self.args.train_directory,
                                   tf_random_seed=self.args.seed)

        if self.args.weighted:
            weight_column = self.WEIGHT_COLUMN
        else:
            weight_column = None

        self.predictor = \
            tf.estimator.DNNClassifier(hidden_units=self.args.hiddens,
                                       feature_columns=self.columns,
                                       n_classes=self.num_labels,
                                       weight_column=weight_column,
                                       config=run_config)

@Model.register('dbn')
class DBNModel(EstimatorModel):
    """
    Deep belief network classifier.
    """

    RUNNER = TFSKLRunner

    def build(self):
        if SupervisedDBNClassification is None:
            raise ImportError('DBN package is not installed')

        self.predictor = SupervisedDBNClassification(
            hidden_layers_structure=[256, 256],
            learning_rate_rbm=0.05,
            learning_rate=self.args.learning_rate,
            n_epochs_rbm=10,
            n_iter_backprop=self.args.num_epochs,
            batch_size=self.args.batch_size,
            activation_function='relu',
            dropout_p=0.2)

class KerasModel(Model):
    """
    Model based on a Keras model.
    """

    RUNNER = KerasRunner

    def __init__(self, args, dtypes, sizes):
        super().__init__(args, dtypes, sizes)
        self.predictor = None

    def build(self):
        raise NotImplementedError('Must be implemented by subclasses')

@Model.register('kdnn')
class KerasDNNModel(KerasModel):
    """
    Deep neural network model from Keras with dropout.
    """

    @classmethod
    def add_arguments(cls, parser):
        group = parser.add_argument_group('KDNN', 'Deep dropout neural network')
        group.add_argument('--dhiddens', nargs='+', type=int,
                           default=[300, 200, 100],
                           help='Number of units per hidden layer')
        group.add_argument('--dropouts', nargs='+', type=float,
                           default=[0.5],
                           help='Ratio of units to drop out on the layers')

    def build(self):
        model = km.Sequential()

        # Add first hidden layer.
        model.add(kl.Dense(self.args.hiddens[0], input_dim=self.num_features,
                           activation='relu'))
        model.add(kl.Dropout(self.args.dropouts[0]))

        # Add other hidden layers
        for index, hiddens in enumerate(self.args.hiddens[1:]):
            model.add(kl.Dense(hiddens, activation='relu'))

            dropout_index = min(index+1, len(self.args.dropouts)-1)
            model.add(kl.Dropout(self.args.dropouts[dropout_index]))

        model.add(kl.Dense(self.num_labels, activation='softmax'))

        if self.num_labels <= 1:
            loss_function = 'binary_crossentropy'
        else:
            loss_function = 'categorical_crossentropy'

        model.compile(optimizer='rmsprop', loss=loss_function,
                      metrics=['accuracy'])

        self.predictor = model
