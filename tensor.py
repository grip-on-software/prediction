"""
Classification and prediction models for sprint features.
"""

from __future__ import print_function
import argparse
import logging
import random
import sys
import numpy as np
import tensorflow as tf
import simplejson
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
    parser.add_argument('--config', default='config.yml',
                        help='File to read store configuration from')
    parser.add_argument('--store', default='local',
                        help='Store type to use to load the file from')
    parser.add_argument('--index', default=None, nargs='*',
                        help='Attribute indexes to use from the dataset')
    parser.add_argument('--remove', default=None, nargs='*',
                        help='Attribute indexes to ignore from the dataset')
    parser.add_argument('--project', default=False, action='store_true',
                        help='Allow distinguishing projects in dataset')
    parser.add_argument('--assign', action='append',
                        help='Attribute to generate from other attributes')
    parser.add_argument('--validation-index', dest='validation_index',
                        default=None, help='Tag index for the validation set')
    parser.add_argument('--combinations', default=False, type=int, nargs='?',
                        const=3, help='Make combinations of number of attributes')
    parser.add_argument('--max-combinations', dest='max_combinations', type=int,
                        default=sys.maxsize,
                        help='Number of combinations to try before giving up')
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
    parser.add_argument('--no-save', default=True, action='store_false',
                        dest='save', help='Do not store model variables')
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
    parser.add_argument('--roll-sprints', dest='roll_sprints', type=int,
                        nargs='?', default=0, const=1,
                        help='Use features of previous sprints')
    parser.add_argument('--keep-index', dest='keep_index', default=None,
                        nargs='*', help='Fatures of current sprint to keep')
    parser.add_argument('--roll-labels', dest='roll_labels', default=False,
                        action='store_true', help='Use labels of previous sprints')
    parser.add_argument('--roll-validation', dest='roll_validation',
                        default=False, action='store_true',
                        help='Use previous features in validation set, shrink train/test')
    parser.add_argument('--keep-incomplete', dest='keep_incomplete',
                        default=False, action='store_true',
                        help='Keep rolled samples that do not have all sprints')
    parser.add_argument('--replace-na', dest='replace_na', type=float,
                        default=False, nargs='?', const=0,
                        help='Replace NaN values with a valid value')
    parser.add_argument('--weighted', default=False, action='store_true',
                        help='Model can use class weights for balancing')
    parser.add_argument('--time', help='Split dataset based on time feature')
    parser.add_argument('--time-size', dest='time_size', type=int,
                        default=10, help='Minimal size of time dataset')
    parser.add_argument('--time-bin', dest='time_bin', type=int,
                        default=10, help='Bin size of time dataset')
    parser.add_argument('--no-stratified-split', dest='stratified_split',
                        default=True, action='store_false',
                        help='Do not create proportionally balanced sets')
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

def alter_json_list(obj):
    """
    Convert a (multidimensional) NumPy array into a (nested) list, with all
    NaN values replaced with None.
    """

    na_indexes = zip(*np.where(obj != np.nan_to_num(obj)))
    output = obj.tolist()
    for na_index in na_indexes:
        part = output
        for index in na_index[:-1]:
            part = part[index]

        part[na_index[-1]] = None

    return output

def serialize_json(obj):
    """
    Serialize an object to a JSON-serializable type when the object is not
    serializable by the default JSON code.
    """

    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, np.ndarray):
        return alter_json_list(obj)
    if isinstance(obj, np.generic):
        return obj.item()

    raise TypeError("Type '{}' is not serializable ({!r})".format(type(obj), obj))

class Classification(object):
    """
    Classification of sprint features.
    """

    def __init__(self, args):
        self.args = args
        self._results = []

    def _export_results(self, data_sets, results):
        predictions = results["labels"]
        if predictions.size == 0:
            logging.warning('No prediction output')
        else:
            data_set = data_sets.data_sets[data_sets.VALIDATION]

            organization_names = np.array(data_sets.organizations)
            keys = {
                "projects": {
                    "key": data_sets.PROJECT_KEY,
                    "filter": lambda projects: projects.astype(np.int)
                },
                "sprints": {
                    "key": data_sets.SPRINT_KEY,
                    "filter": lambda sprints: sprints.astype(np.int)
                },
                "organizations": {
                    "key": data_sets.ORGANIZATION_KEY,
                    "filter": lambda organizations: \
                        organization_names[organizations.astype(np.int)] \
                        if organization_names.size != 0 else None
                }
            }
            validation_context = data_sets.get_context(data_sets.VALIDATION)
            for result_key, result_config in keys.items():
                validation_key = result_config["key"]
                validation = validation_context[:, validation_key]
                results[result_key] = result_config["filter"](validation)

            logging.warning('Projects: %r', results["projects"])
            logging.warning('Sprints: %r', results["sprints"])
            logging.warning('Predicted labels: %r', predictions)
            logging.warning('Actual labels: %r', data_set[data_sets.LABELS])
            results["features"] = data_sets.get_values(data_sets.VALIDATION)
            results["configuration"] = {
                "label": self.args.label,
                "labels": data_sets.labels,
                "assignments": data_sets.assignments,
                "features": data_sets.features,
                "model": self.args.model,
                "binary": self.args.binary,
                "weighted": self.args.weighted,
                "stratified": self.args.stratified_sample
            }

        if self.args.combinations or self.args.time:
            logging.warning('Results: %r', results)
            self._results.append(results)
        else:
            self._write_results(results)

    def _write_results(self, results):
        file_opener = get_file_opener(self.args)
        with file_opener(self.args.results, 'w') as results_file:
            simplejson.dump(results, results_file, default=serialize_json,
                            ignore_nan=True)

    def run_session(self, graph, model, test_ops, data_sets):
        """
        Perform the training epoch runs.
        """

        # Allow CPU operations if no GPU implementation is available
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config, graph=graph) as sess:
            # Create the op for initializing variables.
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())

            sess.run(init_op)

            run_class = model.RUNNER
            runner = run_class(self.args, sess, model, test_ops)
            if self.args.train:
                runner.run(graph, data_sets)

            results = runner.evaluate(graph, data_sets)
            self._export_results(data_sets, results)

    def _clean(self):
        if self.args.clean:
            cleaner = Cleaner(self.args.clean_patterns)
            cleaner.clean(self.args.train_directory)

    def _setup(self):
        logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                            level=getattr(logging, self.args.log.upper(), None))
        logging.getLogger('tensorflow').propagate = False

        np.set_printoptions(threshold=sys.maxsize)
        if self.args.seed is not None:
            random.seed(self.args.seed)
            np.random.seed(self.args.seed)

    def _run(self, data_sets):
        # Tell TensorFlow that the model will be built into the default Graph.
        graph = tf.Graph()
        with graph.as_default():
            with graph.device(self.args.device):
                if self.args.seed is not None:
                    tf.set_random_seed(self.args.seed)

                # Create the training batches, network, and training ops.
                batch_set = data_sets.get_batches(graph, data_sets.TRAIN)
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
                data_sets.get_batches(graph, data_sets.TEST)
                data_sets.get_batches(graph, data_sets.VALIDATION)
                if model.outputs is not None:
                    logging.info('%r', model.outputs)
                    pred = tf.argmax(model.outputs, 1)
                    correct = tf.equal(pred, model.y_labels)
                    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
                else:
                    accuracy = None
                    pred = None

                if not self.args.dry:
                    self.run_session(graph, model, [accuracy, pred], data_sets)

    def main(self, _):
        """
        Main entry point.
        """

        self._clean()
        self._setup()

        data_sets = Dataset(self.args)
        self._run(data_sets)

        if self.args.combinations or self.args.time:
            stop = False
            count = 0
            while count < self.args.max_combinations and not stop:
                try:
                    self._clean()
                    tf.reset_default_graph()
                    data_sets.clear_batches(data_sets.TRAIN)
                    data_sets.clear_batches(data_sets.TEST)
                    data_sets.clear_batches(data_sets.VALIDATION)
                    data_sets.load_datasets()
                    self._run(data_sets)
                    count = count + 1
                except StopIteration:
                    stop = True

            self._write_results(self._results)

def bootstrap():
    """
    Set up the TensorFlow application.
    """

    args, unparsed = get_parser().parse_known_args()
    application = Classification(args)
    tf.app.run(main=application.main, argv=[sys.argv[0]] + unparsed)

if __name__ == "__main__":
    bootstrap()
