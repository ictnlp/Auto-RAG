#!/bin/bash
MODEL_NAME="<name of your deployed vllm model>"
BASE_URL="<your model url>"
ENCODER_PATH='intfloat/e5-base-v2'


CUDA_VISIBLE_DEVICES=4 python Auto-RAG/gui.py \
    model_name=$MODEL_NAME \
    base_url=$BASE_URL \
    database_path=wiki \
    index_type=faiss \
    query_encoder_config.encoder_type=hf \
    query_encoder_config.hf_config.model_path=$ENCODER_PATH
