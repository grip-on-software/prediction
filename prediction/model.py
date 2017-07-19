"""
TensorFlow prediction models.
"""

import math
import tensorflow as tf
from .runner import TFRunner, TFLearnRunner, TFSKLRunner

class Model(object):
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

        pass

    def __init__(self, args, dtypes, sizes):
        self.args = args
        self._num_features, self._num_labels = sizes
        input_dtype, label_dtype = dtypes
        self._x_input = tf.placeholder(dtype=input_dtype,
                                       shape=[None, self._num_features])
        self._y_labels = tf.placeholder(dtype=label_dtype, shape=[None])

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

        return self._x_input

    @property
    def y_labels(self):
        """
        Retrieve the output placeholder variable.
        """

        return self._y_labels

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

@Model.register('mlp')
class MultiLayerPerceptron(Model):
    """
    Neural network model with multiple (visible, hidden, ..., output) layers.
    """

    RUNNER = TFRunner

    def __init__(self, args, dtypes, sizes):
        super(MultiLayerPerceptron, self).__init__(args, dtypes, sizes)
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
            'softmax': tf.nn.sparse_softmax_cross_entropy_with_logits,
            'onehot': tf.nn.sparse_softmax_cross_entropy_with_logits,
            'sigmoid': tf.nn.sigmoid_cross_entropy_with_logits,
            'sigmax': tf.nn.sigmoid_cross_entropy_with_logits
        }
        entropy_func = entropy_funcs[self.args.activation]

        loss = tf.reduce_mean(entropy_func(labels=self.y_labels,
                                           logits=outputs,
                                           name='xentropy'),
                              name='xentropy_mean')

        if self.args.activation == 'onehot':
            self._outputs = tf.one_hot(self.y_labels, self.num_labels)
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
        return optimizer.minimize(loss, global_step=global_step)

class LearnModel(Model):
    """
    Model based on a TensorFlow Learn estimator.
    """

    RUNNER = TFLearnRunner

    def __init__(self, args, dtypes, sizes):
        super(LearnModel, self).__init__(args, dtypes, sizes)
        self.predictor = None

    def build(self):
        raise NotImplementedError('Must be implemented by subclasses')

@Model.register('dnn')
class DNNModel(LearnModel):
    """
    Deep neural network model from TFLearn.
    """

    @classmethod
    def add_arguments(cls, parser):
        group = parser.add_argument_group('DNN', 'Deep neural network')
        group.add_argument('--hiddens', nargs='+', type=int,
                           default=[100, 150, 100],
                           help='Number of units per hidden layer')

    def build(self):
        columns = [
            tf.contrib.layers.real_valued_column("",
                                                 dimension=self.num_features,
                                                 dtype=self.x_input.dtype)
        ]

        run_config = tf.contrib.learn.RunConfig(save_checkpoints_secs=1,
                                                model_dir=self.args.train_directory)

        self.predictor = tf.contrib.learn.DNNClassifier(hidden_units=self.args.hiddens,
                                                        feature_columns=columns,
                                                        n_classes=self.num_labels,
                                                        config=run_config)

@Model.register('dbn')
class DBNModel(LearnModel):
    """
    Deep belief network classifier.
    """

    RUNNER = TFSKLRunner

    def build(self):
        try:
            from dbn.tensorflow import SupervisedDBNClassification
        except ImportError:
            raise

        self.predictor = SupervisedDBNClassification(
            hidden_layers_structure=[256, 256],
            learning_rate_rbm=0.05,
            learning_rate=self.args.learning_rate,
            n_epochs_rbm=10,
            n_iter_backprop=self.args.num_epochs,
            batch_size=self.args.batch_size,
            activation_function='relu',
            dropout_p=0.2)
