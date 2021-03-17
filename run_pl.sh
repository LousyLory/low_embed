#!/bin/bash

# Install newest ptl.
pip install -U git+http://github.com/PyTorchLightning/pytorch-lightning/
# Install example requirements
pip install -r ./requirements.txt

# Download glue data
python3 ./download_glue_data.py --data_dir ./Dataset

export TASK=STS
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

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR
# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

python run_pl_glue.py --data_dir $DATA_DIR \
--task $TASK \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--learning_rate $LEARNING_RATE \
--num_train_epochs $NUM_EPOCHS \
--train_batch_size $BATCH_SIZE \
--seed $SEED \
--do_train \
--do_predict
