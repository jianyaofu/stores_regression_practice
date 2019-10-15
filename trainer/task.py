#!/usr/bin/env python
"""
Runs the training of the simple linear regression model based 
on hyperparameter values received as input parameters.
"""

import argparse
import json
import os

import tensorflow as tf

import trainer.input as input_module
import trainer.model as model

def _get_session_config_from_env_var():
    """
    Returns a tf.ConfigProto instance that has appropriate device_filters set.
    """

    tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))

    if (tf_config and 'task' in tf_config and 'type' in tf_config['task'] and
            'index' in tf_config['task']):
        # Master should only communicate with itself and ps
        if tf_config['task']['type'] == 'master':
            return tf.ConfigProto(device_filters=['/job:ps', '/job:master'])
        # Worker should only communicate with itself and ps
        elif tf_config['task']['type'] == 'worker':
            return tf.ConfigProto(device_filters=[
                '/job:ps',
                '/job:worker/task:%d' % tf_config['task']['index']
            ])
    return None

def train_and_evaluate(args):
    """
    Run the training and evaluate using the high level API.
    """

    def train_input():
        """
        Input function returning batches from the training data set from training.
        """
        return input_module.input_fn(
            args.train_files,
            num_epochs=args.num_epochs,
            batch_size=args.train_batch_size,
            num_parallel_calls=args.num_parallel_calls,
            prefetch_buffer_size=args.prefetch_buffer_size)

    def eval_input():
        """
        Input function returning the entire validation data set for evaluation.
        Shuffling is not required.
        """
        return input_module.input_fn(
            args.eval_files,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_parallel_calls=args.num_parallel_calls,
            prefetch_buffer_size=args.prefetch_buffer_size)

    train_spec = tf.estimator.TrainSpec(
        train_input, max_steps=args.train_steps)

    exporter = tf.estimator.FinalExporter(
        'stores', input_module.SERVING_FUNCTIONS[args.export_format])
    eval_spec = tf.estimator.EvalSpec(
        eval_input,
        steps=args.eval_steps,
        exporters=[exporter],
        name='stores-eval')

    run_config = tf.estimator.RunConfig(
        session_config=_get_session_config_from_env_var())
    run_config = run_config.replace(model_dir=args.job_dir)
    print('Model dir %s' % run_config.model_dir)
    estimator = model.build_estimator(config=run_config)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    # Input Arguments
    PARSER.add_argument(
        '--train-files',
        help='GCS file or local paths to training data',
        nargs='+',
        default='gs://cloud-samples-data/ml-engine/census/data/adult.data.csv')
    PARSER.add_argument(
        '--eval-files',
        help='GCS file or local paths to evaluation data',
        nargs='+',
        default='gs://cloud-samples-data/ml-engine/census/data/adult.test.csv')
    PARSER.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        default='/tmp/census-estimator')
    PARSER.add_argument(
        '--num-parallel-calls',
        help='Number of threads used to read in parallel the training and evaluation',
        type=int)
    PARSER.add_argument(
        '--prefetch_buffer_size',
        help='Naximum number of input elements that will be buffered when prefetching',
        type=int)
    PARSER.add_argument(
        '--num-epochs',
        help="""\
      Maximum number of training data epochs on which to train.
      If both --max-steps and --num-epochs are specified,
      the training job will run for --max-steps or --num-epochs,
      whichever occurs first. If unspecified will run for --max-steps.\
      """,
        type=int)
    PARSER.add_argument(
        '--train-batch-size',
        help='Batch size for training steps',
        type=int,
        default=40)
    PARSER.add_argument(
        '--eval-batch-size',
        help='Batch size for evaluation steps',
        type=int,
        default=40)
    PARSER.add_argument(
        '--train-steps',
        help="""\
      Steps to run the training job for. If --num-epochs is not specified,
      this must be. Otherwise the training job will run indefinitely.""",
        default=100,
        type=int)
    PARSER.add_argument(
        '--eval-steps',
        help='Number of steps to run evalution for at each checkpoint',
        default=100,
        type=int)
    PARSER.add_argument(
        '--export-format',
        help='The input format of the exported SavedModel binary',
        choices=['JSON', 'CSV', 'EXAMPLE'],
        default='JSON')
    PARSER.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')

    ARGUMENTS, _ = PARSER.parse_known_args()

    # Set python level verbosity
    tf.logging.set_verbosity(ARGUMENTS.verbosity)
    # Suppress C++ level warnings.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Run the training job
    train_and_evaluate(ARGUMENTS)