"""
TensorFlow based prediction runners.
"""

import logging
import time
import tensorflow as tf

class Runner(object):
    """
    Train/optimize runner.
    """

    def __init__(self, args, session, model, test_op):
        self.args = args
        self._session = session
        self._model = model
        self._test_op = test_op

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

class TFRunner(Runner):
    """
    Runner for pure TensorFlow models.
    """

    def loop(self, datasets):
        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        inputs, labels = datasets.get_batches(datasets.TRAIN)
        test_inputs, test_labels = datasets.get_batches(datasets.TEST)
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

                if step % self.args.train_interval == 0:
                    self._train_progress(step, batch_feed, train_values, duration)
                if step % self.args.test_interval == 0:
                    test_input, test_label = self._session.run([test_inputs, test_labels])
                    test_feed = {x_input: test_input, y_labels: test_label}
                    accuracy = self._session.run(self._test_op,
                                                 feed_dict=test_feed)
                    logging.info("test accuracy: %f", accuracy)

                    logging.info("saving training state")
                    saver.save(self._session, self.args.train_directory,
                               global_step=step)

                step = step + 1
        except tf.errors.OutOfRangeError:
            logging.info("saving after %d steps", step)
            saver.save(self._session, self.args.train_directory,
                       global_step=step)

    def _train_progress(self, step, batch_feed, train_values, duration):
        logging.info("step %d: loss value %.2f (%.3f sec)", step,
                     train_values[self._model.LOSS_OP], duration)

        summary_str = self._session.run(self._summary_op, feed_dict=batch_feed)
        self._summary_writer.add_summary(summary_str, step)

        accuracy = self._session.run(self._test_op, feed_dict=batch_feed)
        logging.info("train accuracy: %f", accuracy)

class TFLearnRunner(Runner):
    """
    Runner for TensorFlow Learn models.
    """

    def loop(self, datasets):
        def _get_train_input():
            # Enforce new graph
            datasets.clear_batches(datasets.TRAIN)
            return datasets.get_batches(datasets.TRAIN)

        def _get_test_input():
            # Enforce new graph
            datasets.clear_batches(datasets.TEST)
            return datasets.get_batches(datasets.TEST)

        monitor_class = tf.contrib.learn.monitors.ValidationMonitor
        monitor = monitor_class(input_fn=_get_test_input,
                                every_n_steps=self.args.test_interval)

        self._model.predictor.fit(input_fn=_get_train_input,
                                  max_steps=self.args.num_epochs,
                                  monitors=[monitor])

class TFSKLRunner(Runner):
    """
    Runner for models implemented in TensorFlow but imitating scikit-learn.
    """

    def loop(self, datasets):
        datasets.clear_batches(datasets.TRAIN)
        datasets.clear_batches(datasets.TEST)

        self._model.predictor.fit(*datasets.data_sets[datasets.TRAIN])

        # ...
