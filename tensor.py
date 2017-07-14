"""
Tensorflow for sprint features.
"""

from __future__ import print_function
import argparse
import math
import sys
import time
from scipy.io import arff
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

class Runner(object):
    """
    Train/optimize runner.
    """

    def __init__(self, session, model, train_directory, test_op):
        self._session = session
        self._model = model
        self._train_directory = train_directory
        self._test_op = test_op

        # Input enqueue coordinator
        self._coordinator = tf.train.Coordinator()

        # Build the summary operation based on the collection of summaries.
        self._summary_op = tf.summary.merge_all()

        self._summary_writer = tf.summary.FileWriter(self._train_directory,
                                                     self._session.graph)

    def run(self, *datasets):
        """
        Perform the iterative optimization task.
        """

        # Start input enqueue threads.
        threads = tf.train.start_queue_runners(sess=self._session,
                                               coord=self._coordinator)

        try:
            self.loop(*datasets)
        finally:
            # When done, ask the threads to stop.
            self._coordinator.request_stop()

        # Wait for threads to finish.
        self._coordinator.join(threads)

    def loop(self, *datasets):
        """
        Perform the internal loop of iterative training and test accuracy
        reporting.
        """

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        inputs, labels, test_inputs, test_labels = datasets
        x_input = self._model.x_input
        y_labels = self._model.y_labels

        try:
            step = 0
            while not self._coordinator.should_stop():
                start_time = time.time()

                batch_input, batch_labels = self._session.run([inputs, labels])
                batch_feed = {x_input: batch_input, y_labels: batch_labels}
                train_values = self._session.run(self._model.train_ops,
                                                 feed_dict=batch_feed)

                duration = time.time() - start_time

                if step % 100 == 0:
                    self._train_progress(step, batch_feed, train_values, duration)
                if step % 1000 == 0:
                    test_input, test_label = self._session.run([test_inputs, test_labels])
                    test_feed = {x_input: test_input, y_labels: test_label}
                    accuracy = self._session.run(self._test_op,
                                                 feed_dict=test_feed)
                    print("test accuracy: {:.2%}".format(accuracy))

                    print("saving training state")
                    saver.save(self._session, self._train_directory,
                               global_step=step)

                step = step + 1
        except tf.errors.OutOfRangeError:
            print("saving after %d steps" % step)
            saver.save(self._session, self._train_directory, global_step=step)

    def _train_progress(self, step, batch_feed, train_values, duration):
        args = (step, train_values[self._model.LOSS_OP], duration)
        print("step {}: loss value {:.2f} ({:.3f} sec)".format(*args))

        summary_str = self._session.run(self._summary_op, feed_dict=batch_feed)
        self._summary_writer.add_summary(summary_str, step)

        accuracy = self._session.run(self._test_op, feed_dict=batch_feed)
        print("train accuracy: {:.2%}".format(accuracy))

class Model(object):
    """
    A generic prediction/classification model that can be trained/optimized
    through TensorFlow graphs.
    """

    # Index of the loss op.
    LOSS_OP = 1

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

    @classmethod
    def add_arguments(cls, parser):
        group = parser.add_argument_group('MLP', 'Multi-layered perceptron')
        group.add_argument('--num-hidden1', dest='num_hidden1', type=int,
                           default=128, help='Number of units in hidden layer 1')
        group.add_argument('--num-hidden2', dest='num_hidden2', type=int,
                           default=32, help='Number of units in hidden layer 2')

    @staticmethod
    def make_layer(name, inputs, num_visible, num_hidden):
        """
        Make a layer with weights and biases based on the input shapes.
        """

        with tf.name_scope(name):
            stddev = 1.0 / math.sqrt(float(num_visible))
            weights = tf.Variable(tf.truncated_normal([num_visible, num_hidden],
                                                      stddev=stddev),
                                  name='weights')
            biases = tf.Variable(tf.zeros([num_hidden]), name='biases')
            layer = tf.nn.relu(tf.matmul(inputs, weights) + biases)

        return layer

    def build(self):
        num_hidden1 = self.args.num_hidden1
        num_hidden2 = self.args.num_hidden2

        hidden1 = self.make_layer('hidden1', self.x_input, self.num_features,
                                  num_hidden1)
        hidden2 = self.make_layer('hidden2', hidden1, num_hidden1, num_hidden2)
        self._outputs = self.make_layer('softmax_linear', hidden2, num_hidden2,
                                        self.num_labels)

        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_labels,
                                                           logits=self.outputs,
                                                           name='xentropy'),
            name='xentropy_mean')

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

def get_parser():
    """
    Create a parser to parse command line arguments.
    """

    description = 'Perform classification on sprint features'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--filename', default='sprint_features.arff',
                        help='ARFF file to read training data from')
    parser.add_argument('--index', default=None, nargs='*',
                        help='Attribute indexes to use from the dataset')
    parser.add_argument('--remove', default=None, nargs='*',
                        help='Attribute indexes to ignore from the dataset')
    parser.add_argument('--label', default=None,
                        help='Attribute to use instead as label')
    parser.add_argument('--binary', default=None, type=int, nargs='?', const=1,
                        help='Convert label to binary classifications')
    parser.add_argument('--train-directory', dest='train_directory',
                        default='/tmp/data', help='Output of training')
    parser.add_argument('--num-epochs', dest='num_epochs', type=int,
                        default=1000, help='Number of epochs to train')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
                        default=100, help='Size to divide the training set in')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float,
                        default=0.01, help='Initial learning rate')
    parser.add_argument('--num-threads', dest='num_threads', type=int,
                        default=1, help='Number of threads to run the training')
    parser.add_argument('--test-size', dest='test_size', type=float,
                        default=0.20, help='Ratio of dataset to use for test')

    models = Model.get_model_names()
    parser.add_argument('--model', choices=models, default='mlp',
                        help='Prediction model to use')

    for model in models:
        model_class = Model.get_model(model)
        model_class.add_arguments(parser)

    return parser

class Classification(object):
    """
    Classification of sprint features.
    """

    def __init__(self, args):
        self.args = args

    def create_batches(self, train_data, train_labels):
        """
        Create batches of the training input/labels.
        """

        with tf.name_scope('input'):
            # Input data, pin to CPU because rest of pipeline is CPU-only
            with tf.device('/cpu:0'):
                inputs = tf.constant(train_data)
                labels = tf.constant(train_labels)

                inputs, labels = tf.train.slice_input_producer([inputs, labels],
                                                               num_epochs=self.args.num_epochs)
                inputs, labels = tf.train.batch([inputs, labels],
                                                batch_size=self.args.batch_size,
                                                num_threads=self.args.num_threads,
                                                allow_smaller_final_batch=True)

        return [inputs, labels]

    def run_session(self, model, test_op, data_sets):
        """
        Perform the training epoch runs.
        """

        inputs, labels, test_inputs, test_labels = data_sets

        with tf.Session() as sess:
            # Create the op for initializing variables.
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())

            sess.run(init_op)

            runner = Runner(sess, model, self.args.train_directory, test_op)
            runner.run(inputs, labels, test_inputs, test_labels)

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

            print(label_index)
            print(labels)
        else:
            labels = np.random.randint(0, 1+1, size=(full_data.shape[0],))

        print(indexes)
        return indexes, labels

    def get_datasets(self):
        """
        Retrieve the training dataset and labels.
        """

        with open(self.args.filename) as features_file:
            data, meta = arff.loadarff(features_file)

        print(meta)

        full_data = np.array([[cell for cell in row] for row in data],
                             dtype=np.float32)
        indexes, labels = self._select_data(full_data, meta)

        dataset = np.nan_to_num(full_data[:, tuple(indexes)])
        num_labels = max(labels)+1

        train_data, test_data, train_labels, test_labels = \
            train_test_split(dataset, labels, test_size=self.args.test_size)

        return [train_data, train_labels, test_data, test_labels, num_labels]

    def main(self, _):
        """
        Main entry point.
        """

        train_data, train_labels, test_data, test_labels, num_labels = \
            self.get_datasets()

        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():
            # Create the training batches, network, and training ops.
            inputs, labels = self.create_batches(train_data, train_labels)
            num_features = train_data.shape[1]

            model_class = Model.get_model(self.args.model)
            model = model_class(self.args, [inputs.dtype, labels.dtype],
                                [num_features, num_labels])
            model.build()

            # Create the testing batches and test ops.
            test_inputs, test_y = self.create_batches(test_data, test_labels)
            correct = tf.equal(tf.argmax(model.outputs, 1), test_y)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

            self.run_session(model, accuracy,
                             [inputs, labels, test_inputs, test_y])

def bootstrap():
    """
    Set up the TensorFlow application.
    """

    args, unparsed = get_parser().parse_known_args()
    application = Classification(args)
    tf.app.run(main=application.main, argv=[sys.argv[0]] + unparsed)

if __name__ == "__main__":
    bootstrap()
