from datasets import load_dataset
import os
import argparse
import json


url = 'mdwiratathya/SLAKE-vqa-english' # SLAKE
#url = 'flaviagiammarino/vqa-rad' # VQA-RAD

folder = 'SLAKE'
#folder = 'VQA-RAD'

parser = argparse.ArgumentParser()
parser.add_argument('--data_cache_dir', type=str, default='data/data_cache', help='Directory to cache datasets')
parser.add_argument('--folder_path', type=str, default='data/'+folder, help='Directory to save images and jsonl')
parser.add_argument('--split', type=str, default='test', help='Split of the dataset')
parser.add_argument('--image', action='store_true', help='Save images')
parser.add_argument('--qa', action='store_true', help='Save question-answer pairs')
args = parser.parse_args()


data_cache_dir = '/mount/studenten-temp1/users/linchi/2024WS-FM/datasets/data_cache'
ds_vqa_rad = load_dataset(url, cache_dir=data_cache_dir)

print(ds_vqa_rad)

folder_path = '/mount/studenten-temp1/users/linchi/2024WS-FM/datasets/' + folder


# Function to save images
def save_images(dataset, folder_path, split='train'):
    image_folder = os.path.join(folder_path, split)
    os.makedirs(image_folder, exist_ok=True)
    for idx, row in enumerate(dataset):
        # Get the image file and save it
        image_path = os.path.join(image_folder, f"{idx}.png")
        image = row['image']
        image.save(image_path)

# Function to classify question and answer types
def classify_question_answer(row):
    # Infer question_type based on the prompt (extract question from text)

    question = row['question']
    question_type = question.split()[0].upper()
    
    if question_type not in ['WHAT', 'IS', 'DOES', 'WHERE', 'ARE', 'HOW', 'DO']:
        question_type = 'OTHER'
    
    answer = row['answer']
    # Infer answer_type based on the content of the answer (gpt4_answer)
    # If the answer contains "yes" or "no", assume it is a CLOSED type
    if 'yes' in answer.lower() or 'no' in answer.lower():
        answer_type = "CLOSED"
    else:
        answer_type = "OPEN"
    
    return question, question_type, answer, answer_type
# Function to save question-answer pairs to JSONL
def save_qa_to_jsonl(dataset, jsonl_path):
    with open(jsonl_path, 'w') as f:
        f.write("[")
        for idx, row in enumerate(dataset):
            question, question_type, answer, answer_type = classify_question_answer(row)
            data = {
                "question_id": idx,
                "image": f"{idx}.png",
                "text": row["question"],
                "answer": row["answer"],
                "question_type": question_type,
                "answer_type": answer_type,
            }
            # the last line should not have a comma
            if idx != len(dataset) - 1:
                f.write(json.dumps(data) + ",\n")
            else:
                f.write(json.dumps(data) + "]")
        


def main(args):
    if args.image:
        save_images(ds_vqa_rad[args.split], folder_path=folder_path, split=args.split)

    jsonl_file = "question_answer_gt.jsonl"
    jsonl_path = os.path.join(args.folder_path, "_".join([args.split, jsonl_file]))
    if args.qa:
        save_qa_to_jsonl(ds_vqa_rad[args.split], jsonl_path)

    print(f"Images and question-answer pairs saved to {args.folder_path}/{args.split}")

if __name__ == '__main__':
    main(args)