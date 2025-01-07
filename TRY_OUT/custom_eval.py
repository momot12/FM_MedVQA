from bert_score import score  
import pandas as pd
import json 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag


file_pred = 'OUTPUTS_jsonl/tinyllava_test_vqa_rad_answer_pred.jsonl'
df_pred = pd.read_json(file_pred)

file_gt = 'data/VQA-RAD/test_question_answer_gt.jsonl'
df_gt = pd.read_json(file_gt)


def extract_content_words(text):
    """Extracts content words (Nouns, Verbs, Adjectives, Adverbs) from a text."""
    words = word_tokenize(text)  # Tokenize words
    pos_tags = pos_tag(words)  # Get POS tags

    # Keep only content words
    content_words = [word for word, tag in pos_tags if tag.startswith(('N', 'V', 'J', 'R'))]
    
    return ' '.join(content_words)



def bert_score_eval():
    f1_all = []
    results = []

    for (idx, gt), (_, pred) in zip(df_gt.iterrows(), df_pred.iterrows()):
        print(f'\n*** {idx}: ***')
        gt_value = gt['answer'].lower()
        pred_value = pred['answer_pred'].lower()
        
        min_length = min(len(gt_value), len(pred_value))
        gt_words = ""
        pred_words = ""
        
        # YES-NO answers   
        if gt_value.startswith("yes"):
            # yes-yes
            if pred_value.startswith("yes"):
                F1 = [1.0]
            # yes - no
            elif pred_value.startswith("no"):
                F1 = [0.0]
        elif gt_value.startswith("no"):
            # no - no
            if pred_value.startswith("no"):
                F1 = [1.0]
            # no - yes
            elif pred_value.startswith("yes"):
                F1 = [0.0]
                
        # WORDS, SENTENCES answers
        else:
            gt_value_content = extract_content_words(gt_value)
            #pred_value_content = extract_content_words(pred_value)
            
            for word in gt_value_content.lower().split(' '):
                if word.strip() in pred_value.lower():
                    gt_words += f' {word.strip()}'
                    pred_words += f' {word.strip()}'
                else:
                    gt_words = gt_value #_content
                    pred_words = pred_value #_content
            
            # if gt content words were misfiltered before, e.g. gt = one --> gt = " "
            # take original string to compare then     
            if gt_words == " ":
                gt_words = gt_value
                pred_words = pred_value
                
            P, R, F1 = score([pred_words], [gt_words], lang="en", model_type="distilbert-base-uncased", rescale_with_baseline=True)
            F1 = F1.tolist()
        
        # if F1 is negative
        if F1[0] < 0:
            F1 = [0.0]
            
        f1_all.extend(F1)
        
        # Store results for JSON output
        results.append({
            "index": idx,
            "original_gt": gt_value,
            "original_pred": pred_value,
            "gt_words": gt_words,
            "pred_words": pred_words,
            "f1_score": F1, 
        })

    macro_f1 = sum(f1_all) / len(f1_all)
    print(f"Macro F1 Score: {macro_f1:.4f}")

    
    with open('TRY_OUT/3eval_tinyllava_vqarad.json', "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

        
    
bert_score_eval()
  
#Macro F1 Score: 0.3390 -- yes/no as 1.0 or 0.0
#Macro F1 Score: 0.3737 -- negative to 0.0