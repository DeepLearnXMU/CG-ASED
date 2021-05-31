#!/bin/bash
model=xx

python exp.py \
    --cuda \
    --mode test \
    --parser parser_tree \
    --load_model $model \
    --beam_size 5 \
    --test_file data/geo/test.bin \
    --decode_max_time_step 110
