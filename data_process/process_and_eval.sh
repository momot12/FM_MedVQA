# Description: Process the data and evaluate the model

# process from huggingface dataset to our format
# python3 data_process/data_conversion.py --data_cache_dir data/data_cache --folder_path data/VQA-RAD --split test --qa

# process the ground truth and prediction files
# python3 data_process/process_gt.py --input_file data/VQA-RAD/test_question_answer_gt.jsonl --output_file data/VQA-RAD/test_question_answer_gt.jsonl
# python3 data_process/process_answer.py --input_file data/VQA-RAD/test-answer-file.jsonl --output_file data/VQA-RAD/test-answer-processed.jsonl

# evaluate the model
#python3 evaluation/eval.py --gt=output/llava/test_question_answer_gt.jsonl --pred=output/llava/test_vqa_rad_answer-file.jsonl


### TinyLlava ###
# TEST: VQA-RAD__TINY-LLAVA
#python3 evaluation/eval.py --gt=data/VQA-RAD/test_question_answer_gt.jsonl --pred=OUTPUTS_jsonl/tinyllava_test_vqa_rad_answer_pred.jsonl

# TEST: SLAKE__TINY-LLAVA
#python3 evaluation/eval.py --gt=data/SLAKE/test_question_answer_gt.jsonl --pred=OUTPUTS_jsonl/tinyllava_test_slake_answer_pred.jsonl

# TEST: PathVQA__TINY-LLAVA
#python3 evaluation/eval.py --gt=data/PathVQA/test_question_answer_gt.jsonl --pred=OUTPUTS_jsonl/tinyllava_test_pathvqa_answer_pred.jsonl


### Llava-Med ###
# TEST: VQA-RAD__LlavaMed
#python3 evaluation/eval.py --gt=data/VQA-RAD/test_question_answer_gt.jsonl --pred=OUTPUTS_jsonl/llavamed_test_vqa_rad_answer_pred.jsonl

# TEST: SLAKE__LlavaMed ##TODO
#python3 evaluation/eval.py --gt=data/SLAKE/test_question_answer_gt.jsonl --pred=OUTPUTS_jsonl/tinyllava_test_slake_answer_pred.jsonl 

# TEST: PathVQA__LlavaMed ##TODO
#python3 evaluation/eval.py --gt=data/PathVQA/test_question_answer_gt.jsonl --pred=OUTPUTS_jsonl/tinyllava_test_pathvqa_answer_pred.jsonl


### Vilt ###
# TEST: VQA-RAD__Vilt
#echo
#echo 'VQA-RAD'
# python3 evaluation/eval.py --gt=data/VQA-RAD/test_question_answer_gt.jsonl --pred=OUTPUTS_jsonl/vilt_test_vqa_rad_answer_pred.jsonl

# TEST: SLAKE__Vilt
# echo
# echo 'SLAKE'
# python3 evaluation/eval.py --gt=data/SLAKE/test_question_answer_gt.jsonl --pred=OUTPUTS_jsonl/vilt_test_slake_answer_pred.jsonl 

# TEST: PathVQA__Vilt
# echo
# echo 'PathVQA'
# python3 evaluation/eval.py --gt=data/PathVQA/test_question_answer_gt.jsonl --pred=OUTPUTS_jsonl/vilt_test_pathvqa_answer_pred.jsonl


### TinyLlava - Token limit = 1024 ###
# TEST: VQA-RAD__TINY-LLAVA
#python3 evaluation/eval.py --gt=data/VQA-RAD/test_question_answer_gt.jsonl --pred=OUTPUTS_jsonl/1024_tinyllava_test_vqa-rad_answer_pred.jsonl

# TEST: SLAKE__TINY-LLAVA
python3 evaluation/eval.py --gt=data/SLAKE/test_question_answer_gt.jsonl --pred=OUTPUTS_jsonl/1024_tinyllava_test_slake_answer_pred.jsonl