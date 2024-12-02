import json
import argparse
from tqdm import tqdm

# # Function to classify question and answer types
# def classify_question_answer(line):
#     # Infer question_type based on the prompt (extract question from text)
#     question = line.get('text', '').split('\n')[0]  # Extract the question (remove any image-related part)
#     question_type = question.split()[0].upper()
    
#     if question_type not in ['WHAT', 'IS', 'DOES', 'WHERE', 'ARE', 'HOW', 'DO']:
#         question_type = 'OTHER'
    
#     # Infer answer_type based on the content of the answer (gpt4_answer)
#     # If the answer contains "yes" or "no", assume it is a CLOSED type
#     if 'yes' in line['gpt4_answer'].lower() or 'no' in line['gpt4_answer'].lower():
#         answer_type = "CLOSED"
#     else:
#         answer_type = "OPEN"
    
#     return question, question_type, line['gpt4_answer'], answer_type

# Process and convert data to the desired ground truth format
def process_data(input_file, output_file):
    ground_truth_data = []
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = json.load(infile)
        for data in tqdm(lines, desc="Processing"):
            # Load JSON object from line
            # data = json.loads(line.strip())

            # Extract the question, question_type, answer, and answer_type
            # question, question_type, answer, answer_type = classify_question_answer(data)

            # Prepare the dictionary for ground truth
            ground_truth_entry = {
                "id": data["question_id"],  # Use question_id as the ID
                "question": data['text'],
                "answer": data['answer'],
                "question_type": data['question_type'],
                "answer_type": data['answer_type']
            }
            
            # Add to the ground truth list
            ground_truth_data.append(ground_truth_entry)
    
    # Write the ground truth data to the output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Write the processed line to the output JSONL file
        outfile.write("[")
        for idx, processed_data in tqdm(enumerate(ground_truth_data)):
            if idx != len(ground_truth_data) - 1:
                outfile.write(json.dumps(processed_data) + ',\n')
            else:
                outfile.write(json.dumps(processed_data) + ']')


    print(f"Processing complete. Ground truth saved to: {output_file}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process the dataset and convert it to ground truth format.")
    parser.add_argument('--input_file', default='data/VQA-RAD/test_question_answer_gt.jsonl', type=str, help="Path to the input file")
    parser.add_argument('--output_file', default='data/VQA-RAD/test_question_answer_gt.jsonl',  type=str, help="Path to save the ground truth output")

    # Parse the arguments
    args = parser.parse_args()

    # Process the data
    process_data(args.input_file, args.output_file)

if __name__ == '__main__':
    main()
