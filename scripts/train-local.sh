#!/bin/bash

echo "Training local ML model"

MODEL_NAME="stores_lm"
MODEL_DIR=./tmp/trained_models/$MODEL_NAME/
TRAIN_DATA=./data/train_data.csv
EVAL_DATA=./data/eval_data.csv

rm -rf $MODEL_DIR/*

gcloud ai-platform local train \
    --module-name trainer.task \
    --package-path trainer/ \
    --job-dir $MODEL_DIR \
    -- \
    --train-files $TRAIN_DATA \
    --eval-files $EVAL_DATA \
    --train-steps 1000 \
    --eval-steps 100