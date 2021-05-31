#!/bin/bash


test_file="data/conala/test.bin"
model=xx

python exp.py \
    --cuda \
    --mode test \
    --parser parser_tree \
    --load_model $model \
    --beam_size 15 \
    --test_file ${test_file} \
    --evaluator conala_evaluator \
    --decode_max_time_step 100

