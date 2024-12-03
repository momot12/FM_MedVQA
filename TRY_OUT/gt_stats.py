import pandas as pd


path_gt = "/mount/studenten/team-lab-cl/data2024/fm_med_vqa/momo_fm/FM_MedVQA/data/VQA-RAD/test_question_answer_gt.jsonl"

# Read the JSONL file
df = pd.read_json(path_gt)


yes = []
no = []
open_answer = []
for index, row in df.iterrows():

    if row['answer_type'] == 'CLOSED':
        if row['answer'].lower() == 'yes':
            yes.append(row['answer'])
        elif row['answer'].lower() == 'no':
            no.append(row['answer'])
    
    else:
        open_answer.append(row['answer'])
        
print("Ground truth distributions for VQA-RAD test data:")
print(f'# of yes answers: {len(yes)}\t~27%')
print(f'# of yes answers: {len(no)}\t~30%')
print(f'# of open answers: {len(open_answer)}\t~43%')
print(f'Total answers: {len(yes)+len(no)+len(open_answer)}')