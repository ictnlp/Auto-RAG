#!/bin/bash


MODEL_PATH="<your model path>"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel 4 \
    --max-model-len 8192 \
    --port 8888 \
    --host 0.0.0.0
