#!/bin/bash

ENCODER_PATH='intfloat/e5-base-v2'
MODEL_NAME="<name of your deployed vllm model>"
BASE_URL="<your model url>"
DATA_PATH="<path to the test data>"


python -m flexrag.entrypoints.run_assistant \
    user_module=Auto-RAG \
    data_path=$DATA_PATH \
    assistant_type=autorag \
    autorag_config.model_name=$MODEL_NAME \
    autorag_config.base_url=$BASE_URL \
    autorag_config.database_path=wiki \
    autorag_config.index_type=faiss \
    autorag_config.query_encoder_config.encoder_type=hf \
    autorag_config.query_encoder_config.hf_config.model_path=$ENCODER_PATH \
    eval_config.metrics_type=[retrieval_success_rate,generation_f1,generation_em] \
    eval_config.retrieval_success_rate_config.eval_field=text \
    eval_config.response_preprocess.processor_type=[simplify_answer] \
    log_interval=10