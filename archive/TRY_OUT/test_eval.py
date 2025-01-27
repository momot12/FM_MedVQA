from eval import *
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from eval_metrics.glossary import * 


# NOTE: test on vqa-rad, tinyllava

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='vqa-rad', help='Dataset to evaluate on')
parser.add_argument('--use_space', action='store_true', help='Use space to split words')
args = parser.parse_args()
# dataset = "_PathVQA"
dataset = args.dataset
file_pred = f'/mount/studenten-temp1/users/linchi/2024WS-FM/output/llava-med/answer/test-{dataset}_answer-file.jsonl'
df_pred = pd.read_json(file_pred)

file_gt = '/mount/studenten-temp1/users/linchi/2024WS-FM/output/llava-med/test_question_answer_gt.jsonl'
file_pred = 'OUTPUTS_jsonl/tinyllava_test_vqa_rad_answer_pred.jsonl'
df_pred = pd.read_json(file_pred)

file_gt = 'data/VQA-RAD/test_question_answer_gt.jsonl'
df_gt = pd.read_json(file_gt)

# GT: {"id": 0, "question": "is there evidence of an aortic aneurysm?", "answer": "yes", "question_type": "IS", "answer_type": "CLOSED"}
# PR: {"question_id": 0, "question": "is there evidence of an aortic aneurysm?", "question_type": "IS", "answer_pred": "Yes, there is evidence of an aortic aneurysm in"}

# yes=1, no=0
yes_no_dict = {}
open_ans_dict = {}


for (idx, gt), (_, pred) in zip(df_gt.iterrows(), df_pred.iterrows()):

    gt_value = normalize_word(gt['answer'].lower())
    pred_value = normalize_word(pred['answer_pred'].lower())
    
    #gt_value = gt['answer'].lower()
    #pred_value = pred['answer_pred'].lower()
    
    gt_id = gt['id']
    pred_id = pred['question_id']

    # add 1=yes, 0=no, -100=other gt, -200=other pred
    if gt_id == pred_id:
    # for close-ended question (Yes/No)
        if gt_value.startswith('yes'):
            yes_no_dict[gt_id] = [1]
        elif gt_value.startswith('no'):
            yes_no_dict[gt_id] = [0]
        else:
            yes_no_dict[gt_id] = [-100]  
        
        if pred_value.startswith('yes'):
            yes_no_dict[gt_id].append(1)
        elif pred_value.startswith('no'):
            yes_no_dict[gt_id].append(0)
        else:
            yes_no_dict[gt_id].append(-200)
        

        open_ans_dict[gt_id] = [gt_value.split(), pred_value.split()]

precision_all = []
recall_all = []
f1_all = []

y_true = []
y_pred = []

for id, ans in yes_no_dict.items():
    # check that gt and pred BOTH 0 or 1
    # ans[0] = gt, [1] = pred
    if (ans[0] > -1) and (ans[1] > -1):
        y_true.append(ans[0])
        y_pred.append(ans[1])
        
        #precision_all.append(precision_score(y_true=ans[0], y_pred=ans[1]))
        #recall_all.append(recall_score(y_true=ans[0], y_pred=ans[1]))
        #f1_all.append(f1_score(y_true=ans[0], y_pred=ans[1]))
    
    # if one of each has a string instead of yes/no, then count as mismatch
    #if (ans[0] < -1) or (ans[1] < -1):
        # [-100, -200]
     #   if (ans[0] == -100) and (ans[1] == -200):
      #      precision_all.append(precision_score(y_true=0, y_pred=1))
      #      recall_all.append(recall_score(y_true=ans[0], y_pred=ans[1]))
      #      f1_all.append(f1_score(y_true=ans[0], y_pred=ans[1]))
            
    
# BLEU scores
bleu_score_1 = []
bleu_score_2 = []
bleu_score_3 = []
bleu_score_4 = []
   
for id, ans in open_ans_dict.items():
    # [[gt], [pred]]
    # e.g. [['yes'], ['yes,', 'there', 'is', 'evidence', 'of', 'an', 'aortic', 'aneurysm', 'in']]
    reference = ans[0]
    candidate = ans[1]
    
    bleu_score_1.append(sentence_bleu([reference], candidate, weights=(1, 0, 0, 0)))
    bleu_score_2.append(sentence_bleu([reference], candidate, weights=(0, 1, 0, 0)))
    bleu_score_3.append(sentence_bleu([reference], candidate, weights=(0, 0, 1, 0)))
    bleu_score_4.append(sentence_bleu([reference], candidate, weights=(0, 0, 0, 1)))
        
        
precision = precision_score(y_true=y_true, y_pred=y_pred)
recall = recall_score(y_true=y_true, y_pred=y_pred)
f1 = f1_score(y_true=y_true, y_pred=y_pred)



print(f'Number of yes/no answers: {len(y_pred)}')
print(f'\nNumber of yes/no answers: {len(y_pred)}')
print(f'Precision: {precision*100:.2f}%')
print(f'Recall   : {recall*100:.2f}%')
print(f'F1 score : {f1*100:.2f}%')

print(f'\nMean of all answers:')
print(f'Precision: {np.mean(precision_all)*100:.2f}%')
print(f'Recall   : {np.mean(recall_all)*100:.2f}%')
print(f'F1 score : {np.mean(f1_all)*100:.2f}%')

print(f'\nNumber of open answers: {len(open_ans_dict.values())}')
print(f'Mean BLEU-1 Score: {np.mean(bleu_score_1)*100}')
print(f'Mean BLEU-2 Score: {np.mean(bleu_score_2)*100}')
print(f'Mean BLEU-3 Score: {np.mean(bleu_score_3)*100}')
print(f'Mean BLEU-4 Score: {np.mean(bleu_score_4)*100}')