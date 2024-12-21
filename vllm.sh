CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
    --model '/data/yutian/FastChat/240823_strict_filter_wiki6300_nq3700_filter_chinese/25200'\
    --gpu-memory-utilization 0.9 \
    -tp 4 \
    --max-model-len 8192\
    --port 8888\
    --host 0.0.0.0
