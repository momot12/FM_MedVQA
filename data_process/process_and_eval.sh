# Description: Process the data and evaluate the model

# process from huggingface dataset to our format
# python3 data_process/data_conversion.py --data_cache_dir data/data_cache --folder_path data/VQA-RAD --split test --qa

# process the ground truth and prediction files
# python3 data_process/process_gt.py --input_file data/VQA-RAD/test_question_answer_gt.jsonl --output_file data/VQA-RAD/test_question_answer_gt.jsonl
# python3 data_process/process_answer.py --input_file data/VQA-RAD/test-answer-file.jsonl --output_file data/VQA-RAD/test-answer-processed.jsonl

# evaluate the model
python3 evaluation/eval.py --gt=output/llava/test_question_answer_gt.jsonl --pred=output/llava/test_vqa_rad_answer-file.jsonl