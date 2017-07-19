"""
Tensorflow for sprint features.
"""

from __future__ import print_function
import argparse
import logging
import random
import sys
import numpy as np
import tensorflow as tf
from prediction.cleaner import Cleaner
from prediction.dataset import Dataset
from prediction.model import Model

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
    parser.add_argument('--clean', default=False, action='store_true',
                        help='Remove model files from train directory')
    parser.add_argument('--clean-patterns', nargs='*', dest='clean_patterns',
                        default=Cleaner.DEFAULT_PATTERNS,
                        help='Glob patterns to remove from train directory')
    parser.add_argument('--num-epochs', dest='num_epochs', type=int,
                        default=1000, help='Number of epochs to train')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
                        default=100, help='Size to divide the training set in')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float,
                        default=0.01, help='Initial learning rate')
    parser.add_argument('--train-interval', dest='train_interval', type=int,
                        default=100, help='Number of epochs for train progress')
    parser.add_argument('--test-interval', dest='test_interval', type=int,
                        default=1000, help='Number of epochs for test progress')
    parser.add_argument('--num-threads', dest='num_threads', type=int,
                        default=1, help='Number of threads to run the training')
    parser.add_argument('--test-size', dest='test_size', type=float,
                        default=0.20, help='Ratio of dataset to use for test')
    parser.add_argument('--roll-sprints', dest='roll_sprints', default=False,
                        action='store_true', help='Use features of previous sprints')
    parser.add_argument('--seed', type=int, default=None, nargs='?', const=0,
                        help='Set a predefined random seed')
    parser.add_argument('--log', default='WARNING',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Log level (WARNING by default)')

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


    def run_session(self, model, test_ops, data_sets):
        """
        Perform the training epoch runs.
        """

        with tf.Session() as sess:
            # Create the op for initializing variables.
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())

            sess.run(init_op)

            run_class = model.RUNNER
            runner = run_class(self.args, sess, model, test_ops)
            runner.run(data_sets)

    def main(self, _):
        """
        Main entry point.
        """

        if self.args.clean:
            cleaner = Cleaner(self.args.clean_patterns)
            cleaner.clean(self.args.train_directory)

        logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                            level=getattr(logging, self.args.log.upper(), None))

        if self.args.seed is not None:
            random.seed(self.args.seed)
            np.random.seed(self.args.seed)

        data_sets = Dataset(self.args)

        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():
            if self.args.seed is not None:
                tf.set_random_seed(self.args.seed)

            # Create the training batches, network, and training ops.
            inputs, labels = data_sets.get_batches(data_sets.TRAIN)

            model_class = Model.get_model(self.args.model)
            model = model_class(self.args, [inputs.dtype, labels.dtype],
                                [data_sets.num_features, data_sets.num_labels])
            model.build()

            # Create the testing batches and test ops.
            data_sets.get_batches(data_sets.TEST)
            if model.outputs is not None:
                pred = tf.argmax(model.outputs, 1)
                correct = tf.equal(pred, model.y_labels)
                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            else:
                accuracy = None
                pred = None

            self.run_session(model, [accuracy, pred], data_sets)

def bootstrap():
    """
    Set up the TensorFlow application.
    """

    args, unparsed = get_parser().parse_known_args()
    application = Classification(args)
    tf.app.run(main=application.main, argv=[sys.argv[0]] + unparsed)

if __name__ == "__main__":
    bootstrap()
