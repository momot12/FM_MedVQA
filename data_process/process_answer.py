import json
import argparse
from tqdm import tqdm

# Function to classify question and answer types
def classify_question_answer(line):
    # Infer question_type based on the prompt
    question_type = line.get('prompt', '').split()[0].upper()
    
    if question_type not in ['WHAT', 'IS', 'DOES', 'WHERE', 'ARE', 'HOW', 'DO']:
        question_type = 'OTHER'
    
    # Infer answer_type based on the content of the answer (text)
    # If the answer contains "yes" or "no", assume it is a CLOSED type
    if 'yes' in line['text'].lower() or 'no' in line['text'].lower():
        answer_type = "CLOSED"
    else:
        answer_type = "OPEN"
    
    # Update the line with the inferred question_type and answer_type
    line['question_type'] = question_type
    line['answer_type'] = answer_type
    
    return line

# Read the input JSONL file
def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in tqdm(infile, desc="Processing"):
            # Load JSON object from line
            data = json.loads(line.strip())

            # Add question_type and answer_type if they are missing
            if 'question_type' not in data or 'answer_type' not in data:
                processed_data = classify_question_answer(data)
            else:
                processed_data = data

            # Prepare the new structure
            processed_data['question'] = processed_data['prompt']
            processed_data['answer_pred'] = processed_data['text']
            processed_data['question_type'] = processed_data.get('question_type', 'OPEN')
            processed_data['answer_type'] = processed_data.get('answer_type', 'OPEN')
            del processed_data['prompt']
            del processed_data['text']

            # Write the processed line to the output JSONL file
            outfile.write(json.dumps(processed_data) + '\n')

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process a JSONL file and classify question/answer types.")
    parser.add_argument('--input_file', type=str, help="Path to the input JSONL file")
    parser.add_argument('--output_file', type=str, help="Path to the output JSONL file")

    # Parse the arguments
    args = parser.parse_args()

    # Process the JSONL file
    process_jsonl(args.input_file, args.output_file)

    print(f"Processing complete. Output saved to: {args.output_file}")

if __name__ == '__main__':
    main()
