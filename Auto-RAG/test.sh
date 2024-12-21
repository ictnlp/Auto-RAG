set -e
MAIN_MODEL='/data/yutian/FastChat/240823_strict_filter_wiki6300_nq3700_filter_chinese/25200'
MAIN_MODEL_URL=http://10.208.41.162:8888/v1

CUDA_VISIBLE_DEVICES=2,3 python test.py\
    --search_engine adaptive\
    --main_model $MAIN_MODEL \
    --main_model_url $MAIN_MODEL_URL\
    --retrieval_top_k 50\
    --data_path /data/yutian/Auto-RAG/data/flash_rag/nq/test.jsonl\
    --save_path ./output/dense_nq_re5_num3_eli5.jsonl\
    --workers 20\
    --retrieval_max_iter 5\
    --num_passages 3\
    --elicit_max_iter 5\
    --retrieve_mode dense\
    --verbose
