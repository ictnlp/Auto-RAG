import re
import string
import sys
import ujson as json
import re
import string
import itertools
from collections import Counter
import pickle
import os
import jsonlines
import argparse

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def eval_answer(prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    return em, f1, prec, recall


def update_answer(metrics, prediction, golds):
    max_em, max_f1, max_prec, max_recall = 0, 0, 0, 0

    for gold in golds:
        em, f1, prec, recall = eval_answer(prediction, gold)

        max_em = max(max_em, em)
        max_f1 = max(max_f1, f1)
        max_prec = max(max_prec, prec)
        max_recall = max(max_recall, recall)

    metrics['em'] += float(max_em)
    metrics['f1'] += max_f1
    metrics['prec'] += max_prec
    metrics['recall'] += max_recall

    return max_em, max_prec, max_recall



def _eval(prediction_and_gold_file, alias_file):
    aliases = {}

    prediction_and_gold = prediction_and_gold_file

    with open(alias_file) as f:
        for json_line in map(json.loads, f):
            aliases[json_line["Q_id"]] = {
                "aliases": set(json_line["aliases"] + json_line["demonyms"])
            }

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
        'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
        'evi_em': 0, 'evi_f1': 0, 'evi_prec': 0, 'evi_recall': 0,
        'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}

    for id in range(len(prediction_and_gold)):
        gold_answers = {prediction_and_gold[id]['golden_answers'][0]}

        if 'answer_id' in prediction_and_gold[id] and prediction_and_gold[id]['answer_id'] in aliases and aliases[prediction_and_gold[id]['answer_id']]["aliases"]:
            gold_answers.update(aliases[prediction_and_gold[id]['answer_id']]["aliases"])

        answer = prediction_and_gold[id]['trace'].split("Final Answer: ")[-1].split('<|eot_id|>')[0].strip()
        em, prec, recall = update_answer(
            metrics, answer, gold_answers)

    N = len(prediction_and_gold)

    for k in metrics.keys():
        metrics[k] = round(metrics[k] / N * 100, 2)

    print('2wiki', metrics['f1'])
    return metrics['f1']


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()

def isEM(label, output):
    map_dict = {"true": "yes", "false": "no"}
    if label == output:
        return True
    elif label.replace(" ", "") == output.replace(" ", ""):
        return True
    elif label in map_dict:
        if map_dict[label] == output:
            return True
        else:
            return False
    else:
        label = label.split(" ")
        output = output.split(" ")
        if not len(label) == len(output):
            return False
        for l in label:
            if l not in output:
                return False
        return True
    

def nq_eval(data):
    em = 0
    total_num = len(data)
    for id, i in enumerate(data):
        answer = i['trace'].lower().split('Final Answer:'.lower())[-1]

        if '<|eot_id|>' in answer:
            answer = answer.split('<|eot_id|>')[0].strip()
        answer = answer.split('\n')[0].strip()

        if type(i['golden_answers']) == str:
            i['golden_answers'] = eval(i['golden_answers'])

        for j in i['golden_answers']:
            if isEM(normalize_answer(j), normalize_answer(answer)):
                em += 1
                break
    
    return em/total_num


def hotpotqa_eval(data):
    f1 = 0
    total_num = len(data)
    for id, i in enumerate(data):
        answer = i['trace'].lower().split('Final Answer: '.lower())[-1]
        if '<|eot_id|>' in answer:
            answer = answer.split('<|eot_id|>')[0].strip()
        answer = answer.replace('<|start_header_id|>assistant<|end_header_id|>\n', '')

        if type(i['golden_answers']) == str:
            i['golden_answers'] = eval(i['golden_answers'])
        max_f1 = 0
        for j in i['golden_answers']:
            max_f1 = max(max_f1, f1_score(j, answer)[0])
        f1 += max_f1

    return  f1/total_num

def popqa_eval(data):
    f1 = 0
    total_num = len(data)
    for id, i in enumerate(data):
        answer = i['trace'].lower().split('Final Answer: '.lower())[-1]
        if '<|eot_id|>' in answer:
            answer = answer.split('<|eot_id|>')[0].strip()
        answer = answer.replace('<|start_header_id|>assistant<|end_header_id|>\n', '')

        if type(i['golden_answers']) == str:
            i['golden_answers'] = eval(i['golden_answers'])
        max_f1 = 0
        for j in i['golden_answers']:
            max_f1 = max(max_f1, f1_score(j, answer)[0])
        f1 += max_f1

    return f1/total_num

def triviaqa_eval(data):
    em = 0
    total_num = len(data)
    for id, i in enumerate(data):
        answer = i['trace'].lower().split('Final Answer: '.lower())[-1]
        if '<|eot_id|>' in answer:
            answer = answer.split('<|eot_id|>')[0].strip()
        answer = answer.replace('<|start_header_id|>assistant<|end_header_id|>\n', '')

        if type(i['golden_answers']) == str:
            i['golden_answers'] = eval(i['golden_answers'])
        for j in i['golden_answers']:
            if isEM(normalize_answer(j), normalize_answer(answer)):
                em += 1
                break
    return em/total_num

def webqestions_eval(data):
    em = 0
    total_num = len(data)
    for id, i in enumerate(data):
        answer = i['trace'].lower().split('Final Answer: '.lower())[-1]
        if '<|eot_id|>' in answer:
            answer = answer.split('<|eot_id|>')[0].strip()
        answer = answer.replace('<|start_header_id|>assistant<|end_header_id|>\n', '')

        if type(i['golden_answers']) == str:
            i['golden_answers'] = eval(i['golden_answers'])
        for j in i['golden_answers']:
            if isEM(normalize_answer(j), normalize_answer(answer)):
                em += 1
                break
    return em/total_num

def wiki_eval(data):
    return _eval(data, '/data/yutian/question_refine_rag/data/2wiki/id_aliases.json')
    

def mix_eval(data, naive_data, eval_fun, max_iter=10):

    pred = []
    pred_cnt = 0
    naive_cnt = 0
    for i in range(len(data)):
        if 'Retrieved Document_{}'.format(max_iter) in data[i]['trace'] or 'final answer'.lower() not in data[i]['trace'].lower() or '1.1.1.1.1.1' in data[i]['trace']:
            question = data[i]['question']
            for j in range(len(naive_data)):
                if naive_data[j]['question'] == question:
                    pred.append(naive_data[j])
                    naive_cnt += 1
                    break
        else:
            pred.append(data[i])
            pred_cnt += 1
    return eval_fun(pred)

if __name__ == "__main__":
    # 读取数据
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--nq_data_path', type=str, default='/data/yutian/question_refine_rag/data/nq/nq_test_25200_240708_result.jsonl')
    argparser.add_argument('--hotpotqa_data_path', type=str, default='/data/yutian/question_refine_rag/data/hotpotqa/hotpotqa_test_25200_240708_result.jsonl')
    argparser.add_argument('--popqa_data_path', type=str, default='/data/yutian/question_refine_rag/data/popqa/popqa_test_25200_240708_result.jsonl')
    argparser.add_argument('--triviaqa_data_path', type=str, default='/data/yutian/question_refine_rag/data/triviaqa/triviaqa_test_25200_240708_result.jsonl')
    argparser.add_argument('--webqestions_data_path', type=str, default='/data/yutian/question_refine_rag/data/webqestions/webqestions_test_25200_240708_result.jsonl')
    argparser.add_argument('--wiki_data_path', type=str, default='/data/yutian/question_refine_rag/data/2wiki/2wiki_test_25200_240708_result.jsonl')
    argparser.add_argument('--nq_naive_data_path', type=str)
    argparser.add_argument('--hotpotqa_naive_data_path', type=str)
    argparser.add_argument('--popqa_naive_data_path', type=str)
    argparser.add_argument('--triviaqa_naive_data_path', type=str)
    argparser.add_argument('--webqestions_naive_data_path', type=str)
    argparser.add_argument('--wiki_naive_data_path', type=str)
    args = argparser.parse_args()

    with jsonlines.open(args.nq_data_path) as reader:
        nq_data = list(reader)
    with jsonlines.open(args.hotpotqa_data_path) as reader:
        hotpotqa_data = list(reader)
    with jsonlines.open(args.popqa_data_path) as reader:
        popqa_data = list(reader)
    with jsonlines.open(args.triviaqa_data_path) as reader:
        triviaqa_data = list(reader)
    with jsonlines.open(args.webqestions_data_path) as reader:
        webqestions_data = list(reader)
    with jsonlines.open(args.wiki_data_path) as reader:
        wiki_data = list(reader)

    # nq_eval(nq_data)
    # wiki_eval(wiki_data)
    # triviaqa_eval(triviaqa_data)
    # popqa_eval(popqa_data)
    # hotpotqa_eval(hotpotqa_data)
    # webqestions_eval(webqestions_data)

    with jsonlines.open(args.nq_naive_data_path) as reader:
        nq_naive_data = list(reader)
    with jsonlines.open(args.hotpotqa_naive_data_path) as reader:
        hotpotqa_naive_data = list(reader)
    with jsonlines.open(args.popqa_naive_data_path) as reader:
        popqa_naive_data = list(reader)
    with jsonlines.open(args.triviaqa_naive_data_path) as reader:
        triviaqa_naive_data = list(reader)
    with jsonlines.open(args.webqestions_naive_data_path) as reader:
        webqestions_naive_data = list(reader)
    with jsonlines.open(args.wiki_naive_data_path) as reader:
        wiki_naive_data = list(reader)

    mix_nq = mix_eval(nq_data, nq_naive_data, nq_eval, max_iter=6)
    mix_wiki = mix_eval(wiki_data, wiki_naive_data, wiki_eval, max_iter=15)
    mix_triviaqa = mix_eval(triviaqa_data, triviaqa_naive_data, triviaqa_eval, max_iter=6)
    mix_popqa = mix_eval(popqa_data, popqa_naive_data, popqa_eval, max_iter=7)
    mix_hotpotqa = mix_eval(hotpotqa_data, hotpotqa_naive_data, hotpotqa_eval, max_iter=7)
    mix_webq = mix_eval(webqestions_data, webqestions_naive_data, webqestions_eval, max_iter=1)
    

    result = {'nq_em:': mix_nq, 'wiki_f1:': mix_wiki, 'triviaqa_em:': mix_triviaqa, 'popqa_f1:': mix_popqa, 'hotpotqa_f1:': mix_hotpotqa, 'webq_em:': mix_webq}
    print(json.dumps(result, indent=4))



