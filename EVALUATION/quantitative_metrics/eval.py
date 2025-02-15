"""Compute quantitative metrics: F1, precision, recall, bleu (n=1-4), yes/no accuracy.
   Script borrowed from https://github.com/emmali808/BESTMVQA/blob/master/LLaVA-Med/llava/eval/run_eval.py#L103 """
import argparse
import json
import collections
import numpy as np
import random 
import pandas as pd    
from nltk.translate.bleu_score import sentence_bleu
from eval_metrics.evaluate_metrics import calculate_exactmatch, calculate_f1score, bleu, calculate_appearance_with_normalization
from tabulate import tabulate
from eval_metrics.glossary import *

import warnings
warnings.simplefilter('ignore')

def parse_option():
    parser = argparse.ArgumentParser('Evaluation for Medical VQA model generated outputs', add_help=False)
    parser.add_argument('--gt', type=str, default="gt_sample.json", help='path to groundtruth file', )
    parser.add_argument('--pred', type=str, default="prediction_sample.json", help='path to prediction file', )
    args, unparsed = parser.parse_known_args()
    return args

def load_jsonl(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data 


# output format (json) --> {question_id: int, question: string, answer_pred: string, question_type: OPEN/CLOSED, answer_type: OPEN/CLOSED}
def evaluate(gt, pred):
    # pred: output json file
    #   gt: from given test set   


    exact_hit_scores = []
    closed_hit_scores = []
    bleu_score = []
    bleu_score_1 = []
    bleu_score_2 = []
    bleu_score_3 = []
    precisionList = []
    recallList = []
    f1List = []

    for gt_item, pred_item in zip(gt, pred):
        gt_value = gt_item['answer'].lower()
        # can_value = gt_results[1]['value'].lower()
        pred_value = pred_item['answer_pred'].lower()

        gt_value = normalize_word(gt_value)
        pred_value = normalize_word(pred_value)

        if gt_item['answer_type'] == 'OPEN':
            # for open-ended question
            # if gt_value in pred_value:
            #     hit = 1.0
            # else:
            #     hit = 0.0
            # open_hit_scores['hit'].append(hit)
            exact_hit_scores.append(calculate_exactmatch(pred_value, gt_value))
            # exact_scores['q_id'].append(pred_item['question_id'])


            f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
            f1List.append(f1_score)
            precisionList.append(precision)
            recallList.append(recall)
            # f1_scores['q_id'].append(pred_item['question_id'])

            # if isinstance(f1_scores['hit'][-1], str):
            #     # import pdb; pdb.set_trace()

            b_score = sentence_bleu(references=[str(gt_value).lower().split()],
                                        hypothesis=str(pred_value).lower().split())
            b_score_1 = sentence_bleu(references=[str(gt_value).lower().split()],
                                    hypothesis=str(pred_value).lower().split(), weights=(1, 0, 0, 0))
            b_score_2 = sentence_bleu(references=[str(gt_value).lower().split()],
                                    hypothesis=str(pred_value).lower().split(), weights=(0, 1, 0, 0))
            b_score_3 = sentence_bleu(references=[str(gt_value).lower().split()],
                                    hypothesis=str(pred_value).lower().split(), weights=(0, 0, 1, 0))
            
            # bleu_scores['q_id'].append(pred_item['question_id'])
            bleu_score.append(b_score)
            bleu_score_1.append(b_score_1)
            bleu_score_2.append(b_score_2)
            bleu_score_3.append(b_score_3)

        elif gt_item['answer_type'] == 'CLOSED':
            # for close-ended question (Yes/No)
            # closed_scores['q_id'].append(pred_item['question_id'])
            if 'yes' in pred_value or 'no' in pred_value:
                if gt_value in pred_value:
                    closed_hit_scores.append(1)
                else:
                    closed_hit_scores.append(0)
            else:
                closed_hit_scores.append(0)
    
    closed_score = np.mean(closed_hit_scores) if len(closed_hit_scores) != 0 else 0.0

    num_open, num_close = len(closed_hit_scores), len(closed_hit_scores)
    print(f'num_open {num_open} || num_close {num_close}')

    return tabulate(
        [
            ['exact match score', np.mean(exact_hit_scores)*100], 
            ['f1 score', np.mean(f1List)*100], 
            ['precision', np.mean(precisionList)*100], 
            ['recall', np.mean(recallList)*100], 
            ['bleu_score', np.mean(bleu_score)*100], 
            ['bleu_score_1', np.mean(bleu_score_1)*100], 
            ['bleu_score_2', np.mean(bleu_score_2)*100], 
            ['bleu_score_3', np.mean(bleu_score_3)*100], 
            ['yes/no accuracy', closed_score*100]
        ], 
        headers=['Metric', 'Performance']
    )

if __name__ == '__main__':
    args = parse_option()

    gt = json.load(open(args.gt, 'r'))
    
    # [ {}, ] 

    # print (gt)
    pred = json.load(open(args.pred, 'r'))
    # print ("pred: ", pred)

    gt_ids = [item['id'] for item in gt]
    pred_ids = [item['question_id'] for item in pred]
    num_gt_ids, num_pred_ids = len(gt_ids), len(pred_ids)
    print(f'num_gt_ids: {num_gt_ids} || num_pred_ids: {num_pred_ids}')
    # import pdb; pdb.set_trace()
    assert gt_ids == pred_ids, "please make sure pred and gt are exactly matched"

    # perform evaluation
    results = evaluate(gt, pred)
    print(results)
