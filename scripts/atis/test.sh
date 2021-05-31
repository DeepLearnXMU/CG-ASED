#!/bin/bash

model=xx

python exp.py \
    --cuda \
    --mode test \
    --parser parser_tree \
    --load_model $model \
    --beam_size 5 \
    --test_file data/atis/test.bin \
    --save_decode_to decodes/atis/${model_name}.test.decode \
    --decode_max_time_step 110
