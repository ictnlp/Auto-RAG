#!/bin/bash

set -euo pipefail

DEVICE_ID='[0,1,2,3]'
ENCODER_PATH='intfloat/e5-base-v2'

wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
gunzip psgs_w100.tsv.gz

python -m flexrag.entypoints.prepare_index \
    retriever_type=dense \
    corpus_path=[psgs_w100.tsv] \
    saving_fields=[id,title,text] \
    text_process_pipeline.processor_type=[length_filter] \
    text_process_pipeline.length_filter_config.max_chars=4096 \
    text_process_pipeline.length_filter_config.min_chars=10 \
    text_process_fields=[text] \
    dense_config.database_path=wiki \
    dense_config.encode_fields=[text] \
    dense_config.passage_encoder_config.encoder_type=hf \
    dense_config.passage_encoder_config.hf_config.model_path=$ENCODER_PATH \
    dense_config.passage_encoder_config.hf_config.prompt='query: ' \
    dense_config.passage_encoder_config.hf_config.normalize=True \
    dense_config.passage_encoder_config.hf_config.device_id=$DEVICE_ID \
    dense_config.index_type=faiss \
    dense_config.faiss_config.batch_size=12288 \
    dense_config.faiss_config.log_interval=100000 \
    dense_config.batch_size=5376 \
    dense_config.log_interval=100000
