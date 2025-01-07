from bert_score import score  
import pandas as pd
import openai  
import json 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet

# GT: {"id": 0, "question": "is there evidence of an aortic aneurysm?", "answer": "yes", "question_type": "IS", "answer_type": "CLOSED"}
# PR: {"question_id": 0, "question": "is there evidence of an aortic aneurysm?", "question_type": "IS", "answer_pred": "Yes, there is evidence of an aortic aneurysm in"}

file_pred = 'OUTPUTS_jsonl/tinyllava_test_vqa_rad_answer_pred.jsonl'
df_pred = pd.read_json(file_pred)

file_gt = 'data/VQA-RAD/test_question_answer_gt.jsonl'
df_gt = pd.read_json(file_gt)

"""
# add your key here
openai.api_key = "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"

def evaluate_response(question, prediction, ground_truth):

    prompt = f"
    Compare the predicted answer to the ground truth.
    - Question: {question}
    - Predicted Answer: {prediction}
    - Ground Truth: {ground_truth}

    Score the prediction from 1 to 10 on:
    - Accuracy (1-10): How correct is the answer?
    - Completeness (1-10): Does it fully address the question?
    - Clarity (1-10): Is the answer well-structured and understandable?

    Provide the response in JSON format like this:
    {{
        "Accuracy": X,
        "Completeness": Y,
        "Clarity": Z,
        "Justification": "Brief explanation"
    }}
    "

    response = openai.ChatCompletion.create(
        model= "gpt-4", #"gpt-3.5-turbo", #""gpt-4",
        messages=[{"role": "system", "content": "You are an expert evaluator for medical AI responses."},
                  {"role": "user", "content": prompt}]
    )

    return json.loads(response["choices"][0]["message"]["content"])

# Example usage
question = "is there evidence of an aortic aneurysm?"
prediction = "Yes, there is evidence of an aortic aneurysm in the image."
ground_truth = "Yes, the image shows an aortic aneurysm."

evaluation_result = evaluate_response(question, prediction, ground_truth)
print(evaluation_result)



for (idx, gt), (_, pred) in zip(df_gt.iterrows(), df_pred.iterrows()):
    gt_value = gt['answer'] #.lower()
    pred_value = pred['answer_pred'] #.lower()
    questions = gt['question']
    evaluation_result = evaluate_response(questions, pred_value, gt_value)

    print(json.dumps(evaluation_result, indent=4))
    #json.dump(evaluation_result, file, indent=4)
    #    file.write("\n") 
    break
"""

########## BERT SCORE #####################################################
def extract_content_words(text):
    """Extracts content words (Nouns, Verbs, Adjectives, Adverbs) from a text."""
    words = word_tokenize(text)  # Tokenize words
    pos_tags = pos_tag(words)  # Get POS tags

    # Keep only content words
    content_words = [word for word, tag in pos_tags if tag.startswith(('N', 'V', 'J', 'R'))]
    
    return ' '.join(content_words)


def get_synonyms(word):
    synonyms = set()
    for synset in wordnet.synsets(word):  
        for lemma in synset.lemmas():
            synonyms.add(lemma.name().replace("_", " "))  # Replace underscores with spaces
    return list(synonyms)


def bert_score_eval():
    f1_all = []
    results = []

    for (idx, gt), (_, pred) in zip(df_gt.iterrows(), df_pred.iterrows()):
        print(f'\n*** {idx}: ***')
        gt_value = gt['answer'] #.lower()
        pred_value = pred['answer_pred'] #.lower()
        
        min_length = min(len(gt_value), len(pred_value))
        
        gt_words = ""
        pred_words = ""
               
        if gt_value.lower().startswith(("yes", "no")):
            gt_value_trunc = gt_value[:min_length]
            pred_value_trunc = pred_value[:min_length]
            P, R, F1 = score([pred_value_trunc], [gt_value_trunc], lang="en", rescale_with_baseline=True)
            
        else:
            gt_value_content = extract_content_words(gt_value)
            pred_value_content = extract_content_words(pred_value)
            
            for word in gt_value_content.lower().split(' '):
                word_synonym = get_synonyms(word.strip())
                # print(f'Synonyms: {word_synonym}')
                if word.strip() in pred_value.lower():
                    gt_words += f' {word.strip()}'
                    pred_words += f' {word.strip()}'
                elif word_synonym != []:
                    for synonym in word_synonym:
                        if synonym in pred_value.lower():
                            pred_words += f' {word_synonym}'
                            break
                        else: 
                            gt_words = gt_value_content
                            pred_words = pred_value_content
                else:
                    gt_words = gt_value_content
                    pred_words = pred_value_content
                        
            P, R, F1 = score([pred_words], [gt_words], lang="en", rescale_with_baseline=True)
            
        
        f1_all.append(F1.item())
        # Store results for JSON output
        results.append({
            "index": idx,
            "original_gt": gt_value,
            "original_pred": pred_value,
            "truncated_gt": gt_value[:min_length],
            "truncated_pred": pred_value[:min_length],
            "gt_words": gt_words,
            "pred_words": pred_words,
            "f1_score": F1.item()
        })

    macro_f1 = sum(f1_all) / len(f1_all)
    print(f"Macro F1 Score: {macro_f1:.4f}")
    # VQARAD: Change nothing --> Macro F1 Score: -0.0404
    # VQARAD: Synonyms and tructuation  --> Macro F1 Score: 0.3686, Macro F1 Score: 0.5032
    
    with open('TRY_OUT/eval_vqarad.json', "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    
#bert_score_eval()
  


# TRIALS:
gt1 = "not seen here"
pred1 = "the kidney is located in the upper right portion of the image, near" #"Yes, there is a vertebral fracture in the image."
#pred1 = extract_content_words(predd)

# gt = "one"
# pred = "there are two instances of intussusception in the image."

gt = "5mm"
pred = "the mass is described as large and large in the image."

gt2 = "the diaphragm"
pred2 = "The medium density is located close to the anterior abdominal wall."
 
gt3 = "the 3rd ventricle and the lateral ventricles"
pred3 = "In the CT scan, two ventricles can be seen with calcifications"

gt4 = "regression of left frontal mass"
pred4 = "On the left side of the frontal lobe, there is an ab"

print(extract_content_words('one'))


P, R, F1_1 = score([pred], [gt], lang="en", model_type="distilbert-base-uncased", rescale_with_baseline=True)
print(f'F1_1: {F1_1.tolist()}')

#P, R, F1_2 = score([pred2], [gt2], lang="en", rescale_with_baseline=True)
#print(f'F1_2: {F1_2.tolist()}')

#P, R, F1_3 = score([pred3], [gt3], lang="en", rescale_with_baseline=True)
#print(f'F1_3: {F1_1.tolist()}')

# P, R, F1_4 = score([pred4], [gt4], lang="en", rescale_with_baseline=True)
# print(f'F1_4: {F1_1.tolist()}')

