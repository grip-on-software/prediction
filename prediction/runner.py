"""
TensorFlow based prediction runners.
"""

from functools import partial
import logging
import os
import time
import keras
import numpy as np
import sklearn.metrics
from sklearn.preprocessing import MinMaxScaler
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

    def __init__(self, args, session, model, test_ops):
        super(TFRunner, self).__init__(args, session, model, test_ops)

        # Build the summary operation based on the collection of summaries.
        self._summary_op = tf.summary.merge_all()

        if self._summary_op is None:
            logging.info('No summaries')

        self._indexes_placeholder = tf.placeholder(dtype=tf.int32,
                                                   shape=[self.args.batch_size])

    def _build_feed(self, batch_ops):
        batch_inputs, batch_labels, batch_weights, batch_indexes = \
            self._session.run(batch_ops)

        batch_indexes = np.pad(batch_indexes,
                               (0, self.args.batch_size - len(batch_indexes)),
                               'constant', constant_values=0)

        return {
            self._model.x_input: batch_inputs,
            self._model.y_labels: batch_labels,
            self._model.y_weights: batch_weights,
            self._indexes_placeholder: batch_indexes
        }

    def loop(self, datasets):
        # Create a saver for writing training checkpoints.
        if self.args.save:
            saver = tf.train.Saver(max_to_keep=self.args.num_checkpoints)
        else:
            saver = None

        train_batch_ops = datasets.get_batches(datasets.TRAIN)
        test_batch_ops = datasets.get_batches(datasets.TEST)

        try:
            step = 0
            while not self._coordinator.should_stop():
                start_time = time.time()

                batch_feed, train_values = self._train(train_batch_ops)

                duration = time.time() - start_time

                if step % self.args.train_interval == 0:
                    self._train_progress(step, batch_feed, train_values,
                                         duration)

                if step % self.args.test_interval == 0:
                    test_feed = self._build_feed(test_batch_ops)
                    self._test_progress(saver, step, test_feed)

                step = step + 1
        except tf.errors.OutOfRangeError:
            logging.info("Done after %d steps", step)
            if saver is not None:
                saver.save(self._session, self.args.train_directory,
                           global_step=step)

    def _train(self, train_batch_ops):
        batch_feed = self._build_feed(train_batch_ops)
        train_values = self._session.run(self._model.train_ops,
                                         feed_dict=batch_feed)
        return batch_feed, train_values

    def _train_progress(self, step, batch_feed, train_values, duration):
        batch_size = len(batch_feed[self._model.x_input])
        batch_mean_size = batch_size / float(self.args.batch_size)
        loss = train_values[self._model.LOSS_OP] / batch_mean_size
        logging.info("step %d (%d samples): loss value %.2f (%.3f sec)", step,
                     batch_size, loss, duration)

        if self._summary_op is None:
            summary_str = self._session.run(self._summary_op,
                                            feed_dict=batch_feed)
            writer = tf.summary.FileWriterCache.get(self.args.train_directory)
            writer.add_summary(summary_str, step)

        accuracy = self._session.run(self._test_ops, feed_dict=batch_feed)[0]
        logging.info("train accuracy: %.2f", accuracy)

    def _test_progress(self, saver, step, test_feed):
        self._validate(test_feed)

        if saver is not None:
            logging.info("saving training state")
            saver.save(self._session, self.args.train_directory,
                       global_step=step)

    def _validate(self, test_feed):
        test_label = test_feed[self._model.y_labels]
        accuracy, pred = self._session.run(self._test_ops, feed_dict=test_feed)
        logging.debug('Outputs: %r',
                      self._session.run(self._model.outputs,
                                        feed_dict=test_feed))

        average_arg = 'binary' if self.args.binary else None
        precision = sklearn.metrics.precision_score(test_label, pred,
                                                    average=average_arg)
        recall = sklearn.metrics.recall_score(test_label, pred,
                                              average=average_arg)
        if np.any(precision + recall != 0.0):
            f_score = sklearn.metrics.f1_score(test_label, pred,
                                               average=average_arg)
        else:
            f_score = 0.0

        """
        logging.debug("w1: %r b1: %r",
                      self._model.weights1.eval(), self._model.biases1.eval())
        logging.debug("w2: %r b2: %r",
                      self._model.weights2.eval(), self._model.biases2.eval())
        logging.debug("wm: %r bm: %r",
                      self._model.weights_max.eval(),
                      self._model.biases_max.eval())
        """
        logging.info("real labels: %r", test_label)
        logging.info("predictions: %r", pred)

        logging.info("test accuracy: %r", accuracy)
        logging.info("precision: %r", precision)
        logging.info("recall: %r", recall)
        logging.info("f1: %r", f_score)
        logging.info("confusion:\n%r",
                     sklearn.metrics.confusion_matrix(test_label, pred))

        return pred

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

class FullTrainRunner(TFRunner):
    """
    Runner that provides the full training set to the model, in addition to
    the usual train and test batches.
    """

    def __init__(self, args, session, model, test_ops):
        super(FullTrainRunner, self).__init__(args, session, model, test_ops)
        self._train_inputs = None
        self._train_labels = None

    def loop(self, datasets):
        train = datasets.data_sets[datasets.TRAIN]
        self._train_inputs = train[datasets.INPUTS]
        self._train_labels = train[datasets.LABELS]
        super(FullTrainRunner, self).loop(datasets)

    def _build_feed(self, batch_ops):
        feed_dict = super(FullTrainRunner, self)._build_feed(batch_ops)
        train_inputs = np.delete(self._train_inputs,
                                 feed_dict[self._indexes_placeholder],
                                 axis=0)
        train_labels = np.delete(self._train_labels,
                                 feed_dict[self._indexes_placeholder],
                                 axis=0)

        feed_dict.update({
            self._model.train_inputs: train_inputs,
            self._model.train_labels: train_labels
        })
        return feed_dict

class TFEstimatorRunner(Runner):
    """
    Runner for TensorFlow estimator models.
    """

    def _get_input(self, datasets, data_set):
        # Enforce new graph
        datasets.clear_batches(data_set)
        inputs, labels, weights = datasets.get_batches(data_set)[0:3]
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

        hooks = []
        step = tf.train.StepCounterHook(every_n_steps=self.args.train_interval,
                                        output_dir=self.args.train_directory)
        hooks.append(step)

        # Create a saver for writing training checkpoints.
        if self.args.save:
            cpt = tf.train.CheckpointSaverHook(self.args.train_directory,
                                               save_steps=self.args.test_interval)
            hooks.append(cpt)

        self._model.predictor.train(_get_train_input,
                                    steps=self.args.num_epochs,
                                    hooks=hooks)

    def evaluate(self, datasets):
        def _get_test_input():
            return self._get_input(datasets, datasets.TEST)

        def _get_validation_input():
            return self._get_input(datasets, datasets.VALIDATION)

        try:
            metrics = self._model.predictor.evaluate(_get_test_input)
            logging.info('Metrics: %r', metrics)
        except tf.errors.InvalidArgumentError:
            logging.exception('Could not evaluate test set')
            metrics = None

        samples = self._model.predictor.predict(_get_validation_input)
        outputs = {"class_ids": [], "probabilities": [], "logits": []}
        for sample in samples:
            for key in outputs:
                outputs[key].append(sample[key])

        for key in outputs:
            outputs[key] = np.array(outputs[key])

        logging.info('Outputs: %r', outputs)
        indexes = np.squeeze(outputs["class_ids"])
        probabilities = datasets.choose(indexes, outputs["probabilities"])
        risk = self._scale_logits(outputs["logits"])

        return {
            "labels": indexes,
            "probabilities": probabilities,
            "risks": risk,
            "metrics": metrics
        }

    @staticmethod
    def _scale_logits(logits):
        logits = np.nan_to_num(np.squeeze(logits))
        neg_scaler = MinMaxScaler((0.1, 0.49), copy=True)
        pos_scaler = MinMaxScaler((0.5, 0.99), copy=True)

        neg_scaler.fit(-abs(logits).reshape(-1, 1))
        pos_scaler.fit(abs(logits).reshape(-1, 1))

        scaled = np.zeros(shape=len(logits))
        if np.any(logits < 0):
            scaled[logits < 0] = logits[logits < 0] * neg_scaler.scale_ + neg_scaler.min_
        if np.any(logits >= 0):
            scaled[logits >= 0] = logits[logits >= 0] * pos_scaler.scale_ + pos_scaler.min_

        return scaled

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

class KerasRunner(Runner):
    """
    Runner for models implemented in Keras.
    """

    def loop(self, datasets):
        datasets.clear_batches(datasets.TRAIN)
        datasets.clear_batches(datasets.TEST)

        # Define training data
        train_data = datasets.data_sets[datasets.TRAIN][datasets.INPUTS]
        train_labels = datasets.data_sets[datasets.TRAIN][datasets.LABELS]
        one_hot_labels = keras.utils.to_categorical(train_labels)

        # Define callbacks
        calls = []

        checkpoint_path = os.path.join(self.args.train_directory,
                                       'net.{epoch:02d}-{loss:.2f}.hdf5')
        checkpoint_period = self.args.num_epochs // self.args.num_checkpoints
        calls.append(keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     period=checkpoint_period))

        calls.append(keras.callbacks.TensorBoard(log_dir=self.args.train_directory))

        test_partial = partial(self._test, datasets)
        calls.append(keras.callbacks.LambdaCallback(on_epoch_end=test_partial))

        self._model.predictor.fit(train_data, one_hot_labels,
                                  epochs=self.args.num_epochs,
                                  batch_size=self.args.batch_size,
                                  shuffle=True, callbacks=calls,
                                  verbose=self.args.log == 'DEBUG')

    def _test(self, datasets, epoch, logs): # pylint: disable=unused-argument
        if epoch % self.args.test_interval == 0:
            logging.info('Test evaluation at epoch %d', epoch)
            test_data = datasets.data_sets[datasets.TEST][datasets.INPUTS]
            test_labels = datasets.data_sets[datasets.TEST][datasets.LABELS]
            one_hot_labels = keras.utils.to_categorical(test_labels)

            metrics = self._model.predictor.evaluate(test_data, one_hot_labels,
                                                     batch_size=self.args.batch_size)
            logging.info('%r', zip(self._model.predictor.metrics_names, metrics))

    def evaluate(self, datasets):
        datasets.clear_batches(datasets.VALIDATION)
        inputs = datasets.data_sets[datasets.VALIDATION][0]
        return {
            "labels": self._model.predictor.predict(inputs)
        }
