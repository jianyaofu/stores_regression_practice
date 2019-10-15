#!/usr/bin/env python
"""
Defines the format of input data as present in the CSV file.
"""

CSV_COLUMNS = [
    'tv', 'radio', 'newspaper', 'region', 'sales'
]

CSV_COLUMN_DEFAULTS = [[0.0], [0.0], [0.0], [''], [0.0]]

TARGET_COLUMN = 'sales'