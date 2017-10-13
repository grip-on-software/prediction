"""
Tensorflow for sprint features.
"""

from __future__ import print_function
import argparse
import json
import logging
import random
import sys
import numpy as np
import tensorflow as tf
from prediction.cleaner import Cleaner
from prediction.dataset import Dataset
from prediction.files import get_file_opener
from prediction.model import Model

def get_parser():
    """
    Create a parser to parse command line arguments.
    """

    description = 'Perform classification on sprint features'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--filename', default='sprint_features.arff',
                        help='ARFF file to read training data from')
    parser.add_argument('--store', default='local',
                        help='Store type to use to load the file from')
    parser.add_argument('--index', default=None, nargs='*',
                        help='Attribute indexes to use from the dataset')
    parser.add_argument('--remove', default=None, nargs='*',
                        help='Attribute indexes to ignore from the dataset')
    parser.add_argument('--label', default=None,
                        help='Attribute to use instead as label')
    parser.add_argument('--binary', default=None, type=float, nargs='?',
                        const=1, help='Convert label to binary by threshold')
    parser.add_argument('--train-directory', dest='train_directory',
                        default='/tmp/data', help='Output of training')
    parser.add_argument('--num-checkpoints', dest='num_checkpoints',
                        default=100, help='Number of checkpoint models to keep')
    parser.add_argument('--results', default='sprint_labels.json',
                        help='Filename to output JSON results to')
    parser.add_argument('--no-train', default=True, action='store_false',
                        dest='train', help='Skip training, use existing model')
    parser.add_argument('--clean', default=False, action='store_true',
                        help='Remove model files from train directory')
    parser.add_argument('--clean-patterns', nargs='*', dest='clean_patterns',
                        default=Cleaner.DEFAULT_PATTERNS,
                        help='Glob patterns to remove from train directory')
    parser.add_argument('--device', default='/cpu:0',
                        help='TensorFlow device to pin input data to')
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
    parser.add_argument('--roll-labels', dest='roll_labels', default=False,
                        action='store_true', help='Use labels of previous sprints')
    parser.add_argument('--roll-validation', dest='roll_validation',
                        default=False, action='store_true',
                        help='Use previous features in validation set, shrink train/test')
    parser.add_argument('--weighted', default=False, action='store_true',
                        help='Model can use class weights for balancing')
    parser.add_argument('--stratified-sample', dest='stratified_sample',
                        default=False, action='store_true',
                        help='Create proportionally balanced batches')
    parser.add_argument('--seed', type=int, default=None, nargs='?', const=0,
                        help='Set a predefined random seed')
    parser.add_argument('--dry', default=False, action='store_true',
                        help='Do not train or evaluate the model')
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

def serialize_json(obj):
    """
    Serialize an object to a JSON-serializable type when the object is not
    serializable by the default JSON code.
    """

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()

    raise TypeError("Type '{}' is not serializable".format(type(obj)))

class Classification(object):
    """
    Classification of sprint features.
    """

    def __init__(self, args):
        self.args = args

    def _export_results(self, data_sets, results):
        predictions = results["labels"]
        if predictions.size == 0:
            logging.info('No prediction output')
        else:
            data_set = data_sets.data_sets[data_sets.VALIDATION]
            logging.info('Predicted labels: %r', predictions)
            logging.info('Actual labels: %r', data_set[data_sets.LABELS])
            results.update({
                "projects": data_sets.validation_context[:, data_sets.PROJECT_KEY],
                "sprints": data_sets.validation_context[:, data_sets.SPRINT_KEY],
                "configuration": {
                    "label": self.args.label,
                    "model": self.args.model,
                    "binary": self.args.binary,
                    "weighted": self.args.weighted,
                    "stratified": self.args.stratified_sample
                }
            })
        file_opener = get_file_opener(self.args)
        with file_opener(self.args.results, 'w') as results_file:
            json.dump(results, results_file, default=serialize_json)

    def run_session(self, model, test_ops, data_sets):
        """
        Perform the training epoch runs.
        """

        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            # Create the op for initializing variables.
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())

            sess.run(init_op)

            run_class = model.RUNNER
            runner = run_class(self.args, sess, model, test_ops)
            if self.args.train:
                runner.run(data_sets)

            results = runner.evaluate(data_sets)
            self._export_results(data_sets, results)

    def main(self, _):
        """
        Main entry point.
        """

        if self.args.clean:
            cleaner = Cleaner(self.args.clean_patterns)
            cleaner.clean(self.args.train_directory)

        logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                            level=getattr(logging, self.args.log.upper(), None))
        logging.getLogger('tensorflow').propagate = False

        if self.args.seed is not None:
            random.seed(self.args.seed)
            np.random.seed(self.args.seed)

        data_sets = Dataset(self.args)

        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():
            if self.args.seed is not None:
                tf.set_random_seed(self.args.seed)

            # Create the training batches, network, and training ops.
            batch_set = data_sets.get_batches(data_sets.TRAIN)
            inputs = batch_set[data_sets.INPUTS]
            labels = batch_set[data_sets.LABELS]
            weights = batch_set[data_sets.WEIGHTS]

            model_class = Model.get_model(self.args.model)
            model = model_class(self.args,
                                [inputs.dtype, labels.dtype, weights.dtype],
                                [data_sets.num_features, data_sets.num_labels])
            model.build()

            # Create the testing and validation batches and test ops.
            # These batches must be created after the associated graph is
            # created but before the session starts.
            data_sets.get_batches(data_sets.TEST)
            data_sets.get_batches(data_sets.VALIDATION)
            if model.outputs is not None:
                pred = tf.argmax(model.outputs, 1)
                correct = tf.equal(pred, model.y_labels)
                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            else:
                accuracy = None
                pred = None

            if not self.args.dry:
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
