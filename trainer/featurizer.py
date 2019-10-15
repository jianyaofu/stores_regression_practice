#!/usr/bin/env python
"""
Defines the feature columns passed into the model.
"""

import tensorflow as tf

INPUT_COLUMNS = [
    tf.feature_column.categorical_column_with_vocabulary_list(
        'region', ['north', 'south', 'east', 'west']),
    tf.feature_column.numeric_column('tv'),
    tf.feature_column.numeric_column('radio'),
    tf.feature_column.numeric_column('newspaper')
]