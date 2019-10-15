#!/bin/bash

echo "Submitting an AI Platform job"

PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-mlengine
REGION=us-central1
gsutil mb -l $REGION gs://$BUCKET_NAME
gsutil cp -r data gs://$BUCKET_NAME/data
TRAIN_DATA=gs://$BUCKET_NAME/data/train_data.csv
EVAL_DATA=gs://$BUCKET_NAME/data/eval_data.csv
CURRENT_DATE=`date +%Y%m%d_%H%M%S`
MODEL_NAME="stores_lm"
JOB_NAME=${MODEL_NAME}_${CURRENT_DATE}
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME

gcloud ai-platform jobs submit training $JOB_NAME \
    --job-dir $OUTPUT_PATH \
    --runtime-version 1.10 \
    --module-name trainer.task \
    --package-path trainer/ \
    --region $REGION \
    -- \
    --train-files $TRAIN_DATA \
    --eval-files $EVAL_DATA \
    --train-steps 1000 \
    --eval-steps 100 \
    --verbosity DEBUG