"""
Tensorflow for sprint features.
"""

import argparse
import math
import sys
import time
from scipy.io import arff
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

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
    parser.add_argument('--num-threads', dest='num_threads', type=int,
                        default=1, help='Number of threads to run the training')
    parser.add_argument('--num-hidden1', dest='num_hidden1', type=int,
                        default=128, help='Number of units in hidden layer 1')
    parser.add_argument('--num-hidden2', dest='num_hidden2', type=int,
                        default=32, help='Number of units in hidden layer 2')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float,
                        default=0.01, help='Initial learning rate')
    parser.add_argument('--test-size', dest='test_size', type=float,
                        default=0.20, help='Ratio of dataset to use for test')
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

    def make_training(self, loss):
        """
        Set up the training operation.
        """

        tf.summary.scalar('loss', loss)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(self.args.learning_rate)
        return optimizer.minimize(loss, global_step=global_step)

    def run_session(self, ops, placeholders, data_sets):
        """
        Perform the training epoch runs.
        """

        train_op, loss, test_op, logits = ops
        x_input, y_labels = placeholders
        inputs, labels, test_inputs, test_labels = data_sets

        with tf.Session() as sess:
            # Create the op for initializing variables.
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())

            sess.run(init_op)

            # Build the summary operation based on the collection of summaries.
            summary_op = tf.summary.merge_all()

            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver()

            summary_writer = tf.summary.FileWriter(self.args.train_directory,
                                                   sess.graph)

            # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                step = 0
                while not coord.should_stop():
                    start_time = time.time()

                    batch_input, batch_labels = sess.run([inputs, labels])
                    batch_feed = {x_input: batch_input, y_labels: batch_labels}
                    loss_value = sess.run([train_op, loss], feed_dict=batch_feed)[1]

                    duration = time.time() - start_time

                    if step % 100 == 0:
                        print("step {}, loss value {:.2f} ({:.3f} sec)".format(step, loss_value, duration))

                        summary_str = sess.run(summary_op, feed_dict=batch_feed)
                        summary_writer.add_summary(summary_str, step)

                        accuracy = sess.run(test_op, feed_dict=batch_feed)
                        print("train accuracy: {:.2%}".format(accuracy))
                    if step % 1000 == 0:
                        test_input, test_label = sess.run([test_inputs, test_labels])
                        test_feed = {x_input: test_input, y_labels: test_label}
                        accuracy = sess.run(test_op, feed_dict=test_feed)
                        print("test accuracy: {:.2%}".format(accuracy))
                        print(sess.run(logits, feed_dict=test_feed))
                        print(test_label)

                        print("saving training state")
                        saver.save(sess, self.args.train_directory,
                                   global_step=step)

                    step = step + 1
            except tf.errors.OutOfRangeError:
                print("saving after %d steps" % step)
                saver.save(sess, self.args.train_directory, global_step=step)
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

            # Wait for threads to finish.
            coord.join(threads)

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

    def get_datasets(self):
        """
        Retrieve the training dataset and labels.
        """

        with open(self.args.filename) as features_file:
            data, meta = arff.loadarff(features_file)

        print(meta)

        full_data = np.array([[cell for cell in row] for row in data],
                             dtype=np.float32)
        name_translation = dict(zip(meta.names(), range(full_data.shape[1])))

        if self.args.index:
            indexes = set(self._translate(self.args.index, name_translation))
        else:
            indexes = set(range(full_data.shape[1]))

        indexes.remove(0)
        if self.args.remove:
            indexes -= set(self._translate(self.args.remove, name_translation))

        if self.args.label:
            label_index = next(self._translate([self.args.label],
                                               name_translation))
            indexes.remove(label_index)
            column = np.nan_to_num(full_data[:, label_index])
            labels = column.astype(int)
            if np.any(column - labels) != 0:
                raise ValueError('Label column {0} has non-round numbers'.format(self.args.label))

            if self.args.binary is not None:
                labels = (labels >= self.args.binary).astype(int)

            print(label_index)
            print(labels)
        else:
            labels = np.random.randint(0, 1+1, size=(data.shape[0],))

        print(indexes)

        data = np.nan_to_num(full_data[:, tuple(indexes)])
        num_labels = max(labels)+1

        train_data, test_data, train_labels, test_labels = \
            train_test_split(data, labels, test_size=self.args.test_size)

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
            num_hidden1 = self.args.num_hidden1
            num_hidden2 = self.args.num_hidden2
            x_input = tf.placeholder(dtype=inputs.dtype,
                                     shape=[None, num_features])
            y_labels = tf.placeholder(dtype=labels.dtype, shape=[None])

            hidden1 = self.make_layer('hidden1', x_input, num_features, num_hidden1)
            hidden2 = self.make_layer('hidden2', hidden1, num_hidden1, num_hidden2)
            logits = self.make_layer('softmax_linear', hidden2, num_hidden2, num_labels)

            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_labels,
                                                               logits=logits,
                                                               name='xentropy'),
                name='xentropy_mean')

            train_op = self.make_training(loss)

            # Create the testing batches and test ops.
            test_inputs, test_y = self.create_batches(test_data, test_labels)
            correct = tf.equal(tf.argmax(logits, 1), test_y)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

            self.run_session([train_op, loss, accuracy, logits],
                             [x_input, y_labels],
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
