from bert_score import score  
import pandas as pd
import openai  
import json  

# GT: {"id": 0, "question": "is there evidence of an aortic aneurysm?", "answer": "yes", "question_type": "IS", "answer_type": "CLOSED"}
# PR: {"question_id": 0, "question": "is there evidence of an aortic aneurysm?", "question_type": "IS", "answer_pred": "Yes, there is evidence of an aortic aneurysm in"}

file_pred = 'OUTPUTS_jsonl/tinyllava_test_vqa_rad_answer_pred.jsonl'
df_pred = pd.read_json(file_pred)

file_gt = 'data/VQA-RAD/test_question_answer_gt.jsonl'
df_gt = pd.read_json(file_gt)


# add your key here
openai.api_key = ""


def evaluate_response(question, prediction, ground_truth):
    prompt = f"""
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
    """

    response = openai.ChatCompletion.create(
        model= "gpt-3.5-turbo", #""gpt-4",
        messages=[{"role": "system", "content": "You are an expert evaluator for medical AI responses."},
                  {"role": "user", "content": prompt}]
    )

    return json.loads(response["choices"][0]["message"]["content"])

# Example usage
question = "is there evidence of an aortic aneurysm?"
prediction = "Yes, there is evidence of an aortic aneurysm in the image."
ground_truth = "Yes, the image shows an aortic aneurysm."

for (idx, gt), (_, pred) in zip(df_gt.iterrows(), df_pred.iterrows()):
    gt_value = gt['answer'] #.lower()
    pred_value = pred['answer_pred'] #.lower()
    question = gt['question']
    evaluation_result = evaluate_response(question, prediction, ground_truth)
    
    
    print(json.dumps(evaluation_result, indent=4))
    #json.dump(evaluation_result, file, indent=4)
    #    file.write("\n") 
    break







def bert_score_eval():
    f1_all = []

    for (idx, gt), (_, pred) in zip(df_gt.iterrows(), df_pred.iterrows()):
        gt_value = gt['answer'].lower()
        pred_value = pred['answer_pred'].lower()
        
        #if gt_value.startswith("yes"):
            
        min_length = min(len(gt_value), len(pred_value))

        # truncation
        gt_value_trunc = gt_value[:min_length]
        pred_value_trunc = pred_value[:min_length] 
        
        #print(f'GT:{gt_value}\nPR:{pred_value}\n')
        #print(f'GT:{gt_value_trunc}\nPR:{pred_value_trunc}\n')
        
        #P, R, F1 = score([pred_value_trunc], [gt_value_trunc], lang="en", rescale_with_baseline=True)
        P, R, F1 = score([pred_value], [gt_value], lang="en", rescale_with_baseline=True)
        
        f1_all.extend(F1.tolist())
        #print(f1_all)
        #print(F1.tolist())

    macro_f1 = sum(f1_all) / len(f1_all)
    print(f"Macro F1 Score: {macro_f1:.4f}") # Macro F1 Score: -0.0404
    