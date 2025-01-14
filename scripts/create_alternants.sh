#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

FILE_PATH=$1

FILE_PATH_DO=${FILE_PATH}/double-object-filtered.csv
FILE_PATH_PO=${FILE_PATH}/prepositional-filtered.csv
OUTPUT_PATH_PO=${FILE_PATH}/alternant_of_pos.csv
OUTPUT_PATH_DO=${FILE_PATH}/alternant_of_dos.csv

python src/create_alternants.py \
    --file_path $FILE_PATH_PO \
    --output_path $OUTPUT_PATH_PO \
    --type PO

python src/create_alternants.py \
    --file_path $FILE_PATH_DO \
    --output_path $OUTPUT_PATH_DO \
    --type DO

python data/datives/remove_ws.py \
    --file_path $OUTPUT_PATH_DO

python data/datives/remove_ws.py \
    --file_path $OUTPUT_PATH_PO

