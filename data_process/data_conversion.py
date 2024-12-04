from datasets import load_dataset
import os
import argparse
import json


# create directory for dataset
new_ds_dir = input("Enter directory name for new dataset: ")
FOLDER = 'data/'+str(new_ds_dir)
os.makedirs(FOLDER, exist_ok=True)
print(f'Created: {os.path.abspath(new_ds_dir)}')


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_cache_dir', type=str, default='data/data_cache', help='Directory to cache datasets')
parser.add_argument('--folder_path', type=str, default=FOLDER, help='Directory to save images and jsonl')
parser.add_argument('--chosen_ds', type=str, help='Choose number: 1-VQA, 2-SLAKE, 3-PathVQA.')
parser.add_argument('--split', type=str, default='test', help='Split of the dataset')
parser.add_argument('--image', action='store_true', help='Save images')
parser.add_argument('--qa', action='store_true', help='Save question-answer pairs')
args = parser.parse_args()


# Paths to download datsets
if args.chosen_ds == '1': # VQA-RAD
    ds =load_dataset("flaviagiammarino/vqa-rad", cache_dir=args.data_cache_dir)
elif args.chosen_ds == '2': # SLAKE
    ds =load_dataset("mdwiratathya/SLAKE-vqa-english", cache_dir=args.data_cache_dir)
elif args.chosen_ds == '3': # PathVQA
    ds =load_dataset("flaviagiammarino/path-vqa", cache_dir=args.data_cache_dir)
    

# Save images
def save_images(dataset, folder_path, split='train'):
    # Create directory of split 
    image_folder = os.path.join(folder_path, split)
    os.makedirs(image_folder, exist_ok=True)
    # Get the image file and save it
    for idx, row in enumerate(dataset):
        image_path = os.path.join(image_folder, f"{idx}.png")
        image = row['image']
        # Convert CMYK to RGB if necessary
        if image.mode == "CMYK":
            image = image.convert("RGB")
        # Save image
        image.save(image_path, format="PNG")
        

# Classify question and answer types
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


# Save question-answer pairs to JSONL
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
        

# Main work
def main(args):
    # Save images for specific split
    if args.image:
        save_images(dataset=ds[args.split], folder_path=args.folder_path, split=args.split)

    # Save formatted jsonl files
    jsonl_file = "question_answer_gt.jsonl"
    jsonl_path = os.path.join(args.folder_path, "_".join([args.split, jsonl_file]))
    if args.qa:
        save_qa_to_jsonl(ds[args.split], jsonl_path)
        
    print(f"Images and question-answer pairs saved to {args.folder_path}/{args.split}")


if __name__ == '__main__':
    main(args)