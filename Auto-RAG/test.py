# import pdb
# pdb.set_trace()
import os
import sys

sys.path.append('~/Auto-RAG/FlashRAG')

import json
import random
import numpy as np
import pandas as pd
from template import Knowledge_Prompt
from openai import OpenAI

from typing import List, Dict, Tuple
import numpy as np
import tqdm
from argparse import ArgumentParser
import concurrent.futures
import jsonlines
from fastchat.model.model_adapter import get_conversation_template
from flashrag.retriever import DenseRetriever


def load_data(data_path):

    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    if args.max_data_size is not None:
        random.seed(2024)
        data = random.sample(data, args.max_data_size)

    return data



def dense_test_thread(main_model, 
                         dense_retriever, 
                         data, 
                         args,
                         id,
                         trace):
    try:
        i = data[id]
        print('Processing id: ', id)

        retrieved_ids = []
        queries = [i['question']]
        template = get_conversation_template('llama-3')
        template.set_system_message('Answer the question by retrieving external knowledge. Extract useful information from each retrieved document. If the information is insufficient or irrelevant, refine your query and search again until you are able to answer the question.')
        template.append_message(template.roles[0], "Question: "+i['question'].strip())

        max_iter = args.retrieval_max_iter
        current_iter = 0

        first_model_output = None

        while max_iter > 0:
            prompt = template.get_prompt()
            if args.verbose:
                print('input', prompt)

            first_model_output = main_model.completions.create(
                model=args.main_model,
                prompt= prompt,
                temperature=0.0,
                max_tokens=200,
                stop=['<|eot_id|>']
            ).choices[0].text.strip()


            if 'Query:'.lower() in first_model_output.lower():
                queries = [first_model_output.split('Query:')[-1].strip()]
                if '<|eot_id|>' in queries[0]:
                    queries = [queries[0].split('<|eot_id|>')[0].strip()]
                current_iter += 1 
            elif 'final answer' in first_model_output.lower():
                template.append_message(template.roles[1], 
                                    first_model_output.split('<|start_header_id|>assistant<|end_header_id|>')[1].split('<|eot_id|>')[0].strip())
                break
            else:
                print('Exception: Follow Failed')
                print(template.get_prompt())
                print(first_model_output)

            template.append_message(template.roles[1], 
                                    first_model_output.split('<|start_header_id|>assistant<|end_header_id|>')[1].split('<|eot_id|>')[0].strip())

            document = None
            
            queries[0] = queries[0].replace('[Dense]', '').strip()

            documents = []
            retrieval_results = dense_retriever.search(queries[0])

            for result in retrieval_results:
                if result['id'] not in retrieved_ids:
                    retrieved_ids.append(result['id'])
                    documents.append(result['contents'].split('\n')[-1])
                if len(documents) >= args.num_passages:
                    break
            document = ' '.join(documents)
                        

            template.append_message(template.roles[0], 
                                        "Retrieved Document_{}: ".format(current_iter)+document.strip())
            
            max_iter -= 1

        first_model_output=""
        if max_iter == 0:
            first_model_output = main_model.completions.create(
                model=args.main_model,
                prompt= template.get_prompt(),
                temperature=0.0,
                max_tokens=150,
                stop=['<|eot_id|>']
            ).choices[0].text.strip()
            template.append_message(template.roles[1], 
                                        first_model_output.split('<|start_header_id|>assistant<|end_header_id|>')[1].split('<|eot_id|>')[0].strip())

        max_iter = args.elicit_max_iter 
        while 'Refined Query:' in first_model_output and max_iter > 0:
            current_iter+=1
            query = first_model_output.split('Refined Query:')[-1].strip()
            if '<|eot_id|>' in query:
                query = query.split('<|eot_id|>')[0].strip()

            document_prompt = Knowledge_Prompt.format(i['question'], query)

            document = main_model.completions.create(
                model = args.main_model,
                prompt = document_prompt,
                temperature=0.0,
                max_tokens=200,
                stop=['\n', '<|eot_id|>']
            ).choices[0].text.strip()

            template.append_message(template.roles[0], 
                                        "Retrieved Document_{}: ".format(current_iter)+document.strip())
            prompt = template.get_prompt()

            first_model_output = main_model.completions.create(
                model=args.main_model,
                prompt= prompt,
                temperature=0.0,
                max_tokens=150
            ).choices[0].text.strip()

            template.append_message(template.roles[1], 
                                    first_model_output.split('<|start_header_id|>assistant<|end_header_id|>')[1].split('<|eot_id|>')[0].strip())
            max_iter -= 1

        all_output = template.get_prompt()
        if args.verbose:
            print(all_output)

        trace.append({
            'id': id,
            'question': i['question'],
            'trace': all_output,
            'golden_answers': i['golden_answers'],
        })
        if 'answer_id' in i.keys():
            trace[-1]['answer_id'] = i['answer_id']

        return 'success'
    except Exception as e:
        trace.append({
            'id': id,
            'question': i['question'],
            'trace': template.get_prompt(),
            'golden_answers': i['golden_answers'],
        })
        print('Exception!', e)
        return 'fail'


def dense_test(main_model, dense_retriever, data, args):

    trace = []
    futures = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for id in tqdm.trange(len(data)):
            future = executor.submit(
                dense_test_thread, 
                main_model,  
                dense_retriever,
                data, 
                args, 
                id, 
                trace
            )
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An exception occurred: {e}")
    
    with jsonlines.open(args.save_path, 'w') as f:
        f.write_all(trace)
        print('Result Saved!')
    return trace




if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--search_engine', type=str, default='bm25')
    argparser.add_argument('--main_model', type=str, default='/data/yutian/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B-Instruct')
    argparser.add_argument('--main_model_url', type=str, default='http://10.208.41.162:8888/v1')
    argparser.add_argument('--retrieval_top_k', type=int, default=50)
    argparser.add_argument('--data_path', type=str, default='./data/2wiki/train.json')
    argparser.add_argument('--verbose', action='store_true', default=False)
    argparser.add_argument('--workers', type=int, default=20)
    argparser.add_argument('--save_path', type=str, default='./output/2wiki/result.jsonl')
    argparser.add_argument('--retrieval_max_iter', type=int, default=10)
    argparser.add_argument('--elicit_max_iter', type=int, default=5)
    argparser.add_argument('--max_data_size', type=int, default=None)
    argparser.add_argument('--num_passages', type=int, default=1)
    argparser.add_argument('--retrieve_mode', type=str, default='rf')
    argparser.add_argument('--max_iter', type=int, default=10)
    argparser.add_argument('--dense_corpus_path', type=str, default='/data/yutian/FlashRAG/indexes/wiki-18.jsonl')
    argparser.add_argument('--dense_index_path', type=str, default='/data/yutian/FlashRAG/indexes/e5_Flat.index')
    args = argparser.parse_args()


    print('loading dense retriever')
    retrieval_config = {
        'retrieval_method': 'e5',
        'retrieval_model_path': '/data/yutian/FlashRAG/models/e5-base-v2',
        'retrieval_query_max_length': 256,
        'retrieval_use_fp16': True,
        'retrieval_topk': 50,
        'retrieval_batch_size': 32,
        'index_path': args.dense_index_path,
        'corpus_path': args.dense_corpus_path,
        'save_retrieval_cache': False,
        'use_retrieval_cache': False,
        'retrieval_cache_path': None,
        'use_reranker': False,
        'faiss_gpu': False,
        'use_sentence_transformer': False,
        'retrieval_pooling_method': 'mean'
    }

    retriever = DenseRetriever(retrieval_config)
  

    data = load_data(args.data_path)

    # Initialize OpenAI API

    main_model = OpenAI(
        base_url=args.main_model_url,
        api_key="EMPTY",
    )

    rewrite_model = OpenAI(
        base_url=args.rewrite_model_url,
        api_key="EMPTY",
    )

    if args.retrieve_mode == 'dense':
        trace = dense_test(main_model, retriever, data, args)


    