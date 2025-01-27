import pandas as pd


# for GROUND TRUTH
def gt_distribution(df_gt):

    yes = []
    no = []
    open_answer = []
    for index, row in df_gt.iterrows():

        if row['answer_type'] == 'CLOSED':
            if row['answer'].lower() == 'yes':
                yes.append(row['answer'])
            elif row['answer'].lower() == 'no':
                no.append(row['answer'])
        
        else:
            open_answer.append(row['answer'])
    
    total = len(yes)+len(no)+len(open_answer)  
    yes_percent = f'~{(len(yes)/total)*100:.2f}%'
    no_percent = f'~{(len(no)/total)*100:.2f}%'
    open_percent = f'~{(len(open_answer)/total)*100:.2f}%'
            
    print("Ground truth distributions for VQA-RAD test data:")
    print(f'# of yes answers: {len(yes)}\t{yes_percent}')
    print(f'# of yes answers: {len(no)}\t{no_percent}')
    print(f'# of open answers: {len(open_answer)}\t{open_percent}')
    print(f'Total answers: {total}\n')
    
    
# for PRED
def pred_distribution(df_pred):

    yes = []
    no = []
    open_answer = []
    for index, row in df_pred.iterrows():

        if row['answer_type'] == 'CLOSED':
            if row['answer'].lower() == 'yes':
                yes.append(row['answer'])
            elif row['answer'].lower() == 'no':
                no.append(row['answer'])
        
        else:
            open_answer.append(row['answer'])
    
    total = len(yes)+len(no)+len(open_answer)  
    yes_percent = f'~{(len(yes)/total)*100:.2f}%'
    no_percent = f'~{(len(no)/total)*100:.2f}%'
    open_percent = f'~{(len(open_answer)/total)*100:.2f}%'
            
    print("Ground truth distributions for VQA-RAD test data:")
    print(f'# of yes answers: {len(yes)}\t{yes_percent}')
    print(f'# of yes answers: {len(no)}\t{no_percent}')
    print(f'# of open answers: {len(open_answer)}\t{open_percent}')
    print(f'Total answers: {total}\n')



# GT: vqarad
path_gt_vqarad = "/mount/studenten/team-lab-cl/data2024/fm_med_vqa/momo_fm/FM_MedVQA/data/VQA-RAD/test_question_answer_gt.jsonl"
df_gt_vqarad = pd.read_json(path_gt_vqarad)

#gt_distribution(df_gt=df_gt_vqarad)

# INFERENCE: tiny llava - vqarad
path_out = "/mount/studenten/team-lab-cl/data2024/fm_med_vqa/momo_fm/FM_MedVQA/OUTPUTS_jsonl/tinyllava_test_vqa_rad_answer_pred.jsonl"

df_2 = pd.read_json(path_out)

print(df_2)