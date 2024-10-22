import os
import sys


sys.path.append(os.path.expanduser('~/Auto-RAG/FlashRAG'))

import time
import json
import random
import numpy as np
import pandas as pd
from template import Knowledge_Prompt
from openai import OpenAI
import gradio as gr

from typing import List, Dict, Tuple
import numpy as np
import tqdm
from argparse import ArgumentParser
from flashrag.retriever import DenseRetriever
from gradio import ChatMessage

argparser = ArgumentParser()
argparser.add_argument('--search_engine', type=str, default='dense')
argparser.add_argument('--main_model', type=str, default='Auto-RAG')
argparser.add_argument('--main_model_url', type=str, default='')
argparser.add_argument('--rewrite_model', type=str, default='Qwen1.5-32B-Chat')
argparser.add_argument('--rewrite_model_url', type=str, default='')
argparser.add_argument('--retrieval_top_k', type=int, default=50)
argparser.add_argument('--verbose', action='store_true', default=False)
argparser.add_argument('--workers', type=int, default=20)
argparser.add_argument('--retrieval_max_iter', type=int, default=5)
argparser.add_argument('--elicit_max_iter', type=int, default=5)
argparser.add_argument('--num_passages', type=int, default=3)
argparser.add_argument('--max_iter', type=int, default=10)
argparser.add_argument('--dense_corpus_path', type=str, default='../FlashRAG/indexes/wiki-18.jsonl')
argparser.add_argument('--dense_index_path', type=str, default='../FlashRAG/indexes/e5_Flat.index')
args = argparser.parse_args()


html_output = """
    <div style="font-weight: bold; font-size: 25px">
        Auto-RAG: Autonomous Retrieval-Augmented Generation for Large Language Models
    </div>
    <div style="font-weight: bold; font-size: 20px">
        Authors: Tian Yu, Shaolei Zhang, and Yang Feng
    </div>
"""

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
        'faiss_gpu': True,
        'use_sentence_transformer': False,
        'retrieval_pooling_method': 'mean'
    }


print('Loading Dense Retriever...')

backup_history = []

dense_retriever = DenseRetriever(retrieval_config)
main_model = OpenAI(
    base_url=args.main_model_url,
    api_key="EMPTY",
)

def user(user_message, history: list):
    history.append(
        ChatMessage(
            role="user", content=user_message
        )
    )
    yield "", history


show_details = True

def dense_test_thread(history, show_details):
    print(history)
    # time.sleep(10)
    input_question=history[-1]['content'].strip()
    history_openai_format = []
    history_openai_format = [{"role": "system", "content": "Answer the question by retrieving external knowledge. Extract useful information from each retrieved document. If the information is insufficient or irrelevant, refine your query and search again until you are able to answer the question."}]
    history_openai_format.append({"role": "user", "content": 'Question: '+input_question})
    if not show_details:
        history.append(
            ChatMessage(
                role="assistant",
                content="Retrieving and reasoning...",
                metadata={'title': '🤖 Auto-RAG'}
            )
        )
        yield history, []

    retrieved_ids = []
    queries = [input_question]

    max_iter = args.retrieval_max_iter
    current_iter = 0

    first_model_output = None

    while max_iter > 0:

        first_model_output = main_model.chat.completions.create(
            model=args.main_model,
            messages=[{'role': i['role'], 'content': i['content']} for i in history_openai_format],
            temperature=0.0,
            max_tokens=200,
            stop=['<|eot_id|>'],
            stream=True
        )
        history_openai_format.append({"role": "assistant", "content": "", 'metadata': {'title': '🤖 Auto-RAG'}})
        history.append(
            ChatMessage(
                role="assistant",
                content="",
                metadata={'title': '🤖 Auto-RAG'}
            )
        )
        for chunk in first_model_output:
            if chunk.choices[0].delta.content is not None:
                history_openai_format[-1]['content'] += chunk.choices[0].delta.content
                history[-1].content += chunk.choices[0].delta.content
                print(chunk.choices[0].delta.content)
                time.sleep(0.005)
                if show_details:
                    yield history, []
    
        first_model_output = history_openai_format[-1]['content']
        if 'Query:'.lower() in history_openai_format[-1]['content'].lower():
            queries = [history_openai_format[-1]['content'].split('Query:')[-1].strip()]
            history_openai_format.append({"role": "user", "content": '', 'metadata': {'title': '🔍︎ **Dense Retriever**'}})

            current_iter += 1 
        elif 'final answer' in first_model_output.lower():
            break
        else:
            print('Exception: Follow Failed')

        document = None
        
        documents = []
        retrieval_results = dense_retriever.search(queries[0])

        for result in retrieval_results:
            if result['id'] not in retrieved_ids:
                retrieved_ids.append(result['id'])
                documents.append(result['contents'].split('\n')[-1])
            if len(documents) >= args.num_passages:
                break
        document = ' '.join(documents)
                    

        # template.append_message(template.roles[0], 
        #                             "Retrieved Document_{}: ".format(current_iter)+document.strip())
        history_openai_format[-1]['content']='Retrieved Document_{}: '.format(current_iter)+document.strip()
        history.append(
            ChatMessage(
                role="assistant",
                content="Retrieved Document_{}: ".format(current_iter)+document.strip(),
                metadata={'title': '🔍︎ Dense Retriever'}
            )
        )
        if show_details:
            yield history, []

        max_iter -= 1

    first_model_output=""
    if max_iter == 0:
        history_openai_format.append({"role": "assistant", "content": "", 'metadata': {'title': '🤖 Auto-RAG'}})
        history.append(
            ChatMessage(
                role="assistant",
                content="",
                metadata={'title': '🤖 Auto-RAG'}
            )
        )
        first_model_output = main_model.chat.completions.create(
            model=args.main_model,
            messages=[{'role': i['role'], 'content': i['content']} for i in history_openai_format],
            temperature=0.0,
            max_tokens=150,
            stop=['<|eot_id|>'],
            stream=True
        )
        for chunk in first_model_output:
            if chunk.choices[0].delta.content is not None:
                history_openai_format[-1]['content'] += chunk.choices[0].delta.content
                history[-1].content += chunk.choices[0].delta.content
                if show_details:
                    yield history, []
        first_model_output = history_openai_format[-1]['content']
    max_iter = args.elicit_max_iter 
    while 'Refined Query:' in history_openai_format[-1]['content'] and max_iter > 0:
        current_iter+=1
        query = history_openai_format[-1]['content'].split('Refined Query:')[-1].strip()

        document_prompt = Knowledge_Prompt.format(input_question, query)

        history_openai_format.append({"role": "user", "content": 'Eliciting Paramatric Knowledge for Query: '+query, 'metadata': {'title': 'Parametric Knowledge'}})
        history.append(
            ChatMessage(
                role="user",
                content='',
                metadata={'title': 'Parametric Knowledge'}
            )
        )
        if show_details:
            yield history, []


        document = main_model.completions.create(
            model = args.main_model,
            prompt = document_prompt,
            temperature=0.0,
            max_tokens=200,
            stop=['\n', '<|eot_id|>'],
        ).choices[0].text.strip()

        history_openai_format[-1]['content']='Retrieved Document_{}: '.format(current_iter)+document.strip()
        history[-1].content='Retrieved Document_{}: '.format(current_iter)+document.strip()
        if show_details:
            yield history

        print(history_openai_format)
        history_openai_format.append({"role": "assistant", "content": "", 'metadata': {'title': '🤖 Auto-RAG'}})
        history.append(
            ChatMessage(
                role="assistant",
                content="",
                metadata={'title': '🤖 Auto-RAG'}
            )
        )
        first_model_output = main_model.chat.completions.create(
            model=args.main_model,
            messages= history_openai_format,
            temperature=0.0,
            max_tokens=150,
            stop=['<|eot_id|>'],
            stream=True
        )

        

        for chunk in first_model_output:
            if chunk.choices[0].delta.content is not None:
                history_openai_format[-1]['content'] += chunk.choices[0].delta.content
                history[-1].content += chunk.choices[0].delta.content
                if show_details:
                    yield history, []

        max_iter -= 1

    if not show_details:
        # backup_history = history
        # history = [history[0], history[-1]]
        # history[-1].content = history[-1].content.split('Final Answer:')[-1].strip()
        backup_history = []
        for id in range(len(history)):
            print(history[id])
            new_item = {}
            if type(history[id]) == dict:
                new_item['role'] = history[id]['role']
                new_item['content'] = history[id]['content']
                if 'metadata' in history[id]:
                    new_item['metadata'] = history[id]['metadata']
            else:
                new_item['role'] = history[id].role
                new_item['content'] = history[id].content
                if history[id].metadata:
                    new_item['metadata'] = history[id].metadata
            backup_history.append(ChatMessage(**new_item))
        history = [history[0], history[-1]]
        history[-1].content = history[-1].content.split('Final Answer:')[-1].strip()
    else:
        # backup_history = [history[0], history[-1]]
        # backup_history[-1].content = backup_history[-1].content.split('Final Answer:')[-1].strip()
        backup_history = []
        backup_history.append(history[0])
        backup_history.append({'role': history[-1].role, 'content': history[-1].content.split('Final Answer:')[-1].strip(), 'metadata': history[-1].metadata})


    print(history)
    print(backup_history)
    yield history, backup_history


def update_details_button(show_details):
    new_label = "Hide Details" if show_details else "Show Details"
    return gr.update(value=new_label)

def update_show_details(show_details):

    show_details = not show_details
    return show_details

def update_history(history, backup_history):
    print(history)
    print(backup_history)
    tmp = history
    history = backup_history
    backup_history = tmp
    return history, backup_history

def print_backup_history(backup_history):
    print(backup_history)
    return backup_history

with gr.Blocks() as demo:
    gr.HTML(html_output)
    show_details = gr.State(True)
    backup_history = gr.State([])

    chatbot = gr.Chatbot(type="messages", label="Auto-RAG", height=500, placeholder='Ask me anything!', show_copy_button=True, bubble_full_width=False, layout="bubble")
    msg = gr.Textbox()
    with gr.Row():
        toggle_button = gr.Button(f"Hide Details")
        # submit_button = gr.Button("Submit")
        clear_button = gr.Button("Clear")
    toggle_button.click(update_show_details, show_details, show_details).then(update_details_button, show_details, toggle_button).then(update_history, [chatbot, backup_history], [chatbot, backup_history])
    msg.submit(user, [msg, chatbot], [msg,chatbot], queue=False).then(dense_test_thread, [chatbot, show_details], [chatbot, backup_history])
    # submit_button.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(dense_test_thread, [chatbot, show_details], [chatbot, backup_history]).then(print_backup_history, backup_history, backup_history)
    clear_button.click( lambda x:[], chatbot, chatbot).then(lambda x: [], backup_history, backup_history)
demo.launch(server_name="0.0.0.0")