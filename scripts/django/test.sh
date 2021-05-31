#!/bin/bash

test_file="data/django/test.bin"
load_model=xx

python exp.py \
    --cuda \
    --mode test \
    --parser parser_tree \
    --load_model $load_model \
    --beam_size 15 \
    --test_file data/django/test.bin \
    --save_decode_to decodes/django/test.decode \
    --decode_max_time_step 100
