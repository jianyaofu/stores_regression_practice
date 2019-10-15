#!/usr/bin/env python
"""
Defines the simple linear regression model
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import trainer.featurizer as featurizer

def build_estimator(config):
    """
    Args:
        config: (tf.contrib.learn.RunConfig) defining the runtime environment for
            the estimator (including model_dir).
    Returns:
        A LinearRegressor
    """
    input_columns = featurizer.INPUT_COLUMNS
    
    return tf.estimator.LinearRegressor(
        config=config,
        feature_columns=input_columns
    )