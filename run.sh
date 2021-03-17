#!/bin/bash

export TASK_NAME=STS-B
export DATA_DIR=./Dataset/STS-B
export MAX_LENGTH=128
export LEARNING_RATE=2e-5
export BERT_MODEL=bert-base-uncased
export BATCH_SIZE=32
export NUM_EPOCHS=3
export SEED=2
export OUTPUT_DIR_NAME=outputs
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

python run_glue.py \
  --model_name_or_path $BERT_MODEL \
  --cache_dir Dataset/models \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_predict \
  --data_dir $DATA_DIR \
  --max_seq_length $MAX_LENGTH \
  --per_device_train_batch_size $BATCH_SIZE \
  --seed $SEED \
  --learning_rate $LEARNING_RATE \
  --num_train_epochs $NUM_EPOCHS \
  --output_dir $OUTPUT_DIR \
  --overwrite_output_dir
