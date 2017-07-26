"""
TensorFlow based prediction runners.
"""

import logging
import time
import numpy as np
import sklearn.metrics
import tensorflow as tf

class Runner(object):
    """
    Train/optimize runner.
    """

    def __init__(self, args, session, model, test_ops):
        self.args = args
        self._session = session
        self._model = model
        self._test_ops = test_ops

        # Input enqueue coordinator
        self._coordinator = tf.train.Coordinator()

        # Build the summary operation based on the collection of summaries.
        self._summary_op = tf.summary.merge_all()

        self._summary_writer = tf.summary.FileWriter(self.args.train_directory,
                                                     self._session.graph)

    def run(self, datasets):
        """
        Perform the iterative optimization task.
        """

        # Start input enqueue threads.
        threads = tf.train.start_queue_runners(sess=self._session,
                                               coord=self._coordinator)

        try:
            self.loop(datasets)
        finally:
            # When done, ask the threads to stop.
            self._coordinator.request_stop()

        # Wait for threads to finish.
        self._coordinator.join(threads)

    def loop(self, datasets):
        """
        Perform the internal loop of iterative training and test accuracy
        reporting.
        """

        raise NotImplementedError('Must be implemented by subclasses')

    def evaluate(self, datasets):
        """
        Calculate labels for a validation set. If possible, any metrics about
        this set may be reported as well. Returns a dictionary containing the
        predicted labels in "labels" for the validation set and any additional
        metrics such as probabilities and accuracy. Values may be numpy arrays
        or serializable values.
        """

        raise NotImplementedError('Must be implemented by subclasses')

class TFRunner(Runner):
    """
    Runner for pure TensorFlow models.
    """

    def _build_feed(self, batch_ops):
        batch_inputs, batch_labels, batch_weights = self._session.run(batch_ops)

        return {
            self._model.x_input: batch_inputs,
            self._model.y_labels: batch_labels,
            self._model.y_weights: batch_weights
        }

    def loop(self, datasets):
        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        train_batch_ops = datasets.get_batches(datasets.TRAIN)
        test_batch_ops = datasets.get_batches(datasets.TEST)

        try:
            step = 0
            while not self._coordinator.should_stop():
                start_time = time.time()

                batch_feed = self._build_feed(train_batch_ops)
                train_values = self._session.run(self._model.train_ops,
                                                 feed_dict=batch_feed)

                duration = time.time() - start_time

                if step % self.args.train_interval == 0:
                    self._train_progress(step, batch_feed, train_values,
                                         duration)

                if step % self.args.test_interval == 0:
                    test_feed = self._build_feed(test_batch_ops)
                    self._test_progress(saver, step, test_feed)

                step = step + 1
        except tf.errors.OutOfRangeError:
            logging.info("saving after %d steps", step)
            saver.save(self._session, self.args.train_directory,
                       global_step=step)

    def _train_progress(self, step, batch_feed, train_values, duration):
        batch_size = len(batch_feed[self._model.x_input])
        batch_mean_size = batch_size / float(self.args.batch_size)
        loss = train_values[self._model.LOSS_OP] / batch_mean_size
        logging.info("step %d (%d samples): loss value %.2f (%.3f sec)", step,
                     batch_size, loss, duration)

        summary_str = self._session.run(self._summary_op, feed_dict=batch_feed)
        self._summary_writer.add_summary(summary_str, step)

        accuracy = self._session.run(self._test_ops, feed_dict=batch_feed)[0]
        logging.info("train accuracy: %.2f", accuracy)

    def _test_progress(self, saver, step, test_feed):
        self._validate(test_feed)

        logging.info("saving training state")
        saver.save(self._session, self.args.train_directory, global_step=step)

    def _validate(self, test_feed):
        test_label = test_feed[self._model.y_labels]
        accuracy, pred = self._session.run(self._test_ops, feed_dict=test_feed)
        logging.debug('Outputs: %r',
                      self._session.run(self._model.outputs,
                                        feed_dict=test_feed))

        precision = sklearn.metrics.precision_score(test_label, pred)
        recall = sklearn.metrics.recall_score(test_label, pred)
        if precision + recall != 0.0:
            f_score = sklearn.metrics.f1_score(test_label, pred)
        else:
            f_score = 0.0

        logging.debug("w1: %r b1: %r",
                      self._model.weights1.eval(), self._model.biases1.eval())
        logging.debug("w2: %r b2: %r",
                      self._model.weights2.eval(), self._model.biases2.eval())
        logging.debug("wm: %r bm: %r",
                      self._model.weights_max.eval(),
                      self._model.biases_max.eval())

        logging.info("test accuracy: %.2f", accuracy)
        logging.info("precision: %.2f", precision)
        logging.info("recall: %.2f", recall)
        logging.info("f1: %.2f", f_score)
        logging.info("confusion:\n%r",
                     sklearn.metrics.confusion_matrix(test_label, pred))

        return test_label

    def evaluate(self, datasets):
        validation_batch_ops = datasets.get_batches(datasets.VALIDATION)
        stop = False
        labels = []
        while not stop:
            try:
                validation_batch = self._build_feed(validation_batch_ops)
                labels.append(self._validate(validation_batch))
            except tf.errors.OutOfRangeError:
                stop = True

        return {
            "labels": np.hstack(labels)
        }

class TFLearnRunner(Runner):
    """
    Runner for TensorFlow Learn models.
    """

    def _get_input(self, datasets, data_set):
        # Enforce new graph
        datasets.clear_batches(data_set)
        inputs, labels, weights = datasets.get_batches(data_set)
        input_columns = {
            self._model.INPUT_COLUMN: inputs,
            self._model.WEIGHT_COLUMN: weights
        }
        return input_columns, tf.reshape(labels, shape=(-1, 1))

    def loop(self, datasets):
        def _get_train_input():
            return self._get_input(datasets, datasets.TRAIN)

        def _get_test_input():
            return self._get_input(datasets, datasets.TEST)

        monitor_class = tf.contrib.learn.monitors.ValidationMonitor
        monitor = monitor_class(input_fn=_get_test_input,
                                every_n_steps=self.args.test_interval)

        self._model.predictor.fit(input_fn=_get_train_input,
                                  steps=self.args.num_epochs,
                                  monitors=[monitor])

    def evaluate(self, datasets):
        def _get_validation_input():
            return self._get_input(datasets, datasets.VALIDATION)

        probabilities = \
            self._model.predictor.predict_proba(input_fn=_get_validation_input,
                                                as_iterable=False)

        classes = np.argmax(probabilities, axis=1)
        return {
            "labels": classes,
            "probabilities": np.choose(classes, probabilities.transpose())
        }

class TFSKLRunner(Runner):
    """
    Runner for models implemented in TensorFlow but imitating scikit-learn.
    """

    def loop(self, datasets):
        datasets.clear_batches(datasets.TRAIN)
        datasets.clear_batches(datasets.TEST)

        train_data = datasets.data_sets[datasets.TRAIN][datasets.INPUTS]
        train_labels = datasets.data_sets[datasets.TRAIN][datasets.LABELS]
        self._model.predictor.fit(train_data, train_labels)

    def evaluate(self, datasets):
        datasets.clear_batches(datasets.VALIDATION)
        inputs = datasets.data_sets[datasets.VALIDATION][0]
        return {
            "labels": self._model.predictor.predict(inputs)
        }
