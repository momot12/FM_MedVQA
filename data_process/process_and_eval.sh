# Description: Process the data and evaluate the model

# process from huggingface dataset to our format
# python3 data_process/data_conversion.py --data_cache_dir data/data_cache --folder_path data/VQA-RAD --split test --qa

# process the answwer file (only LLaVA-Med)
for dataset in "PathVQA" "SLAKE" "VQA-RAD"
do 
    # echo "Processing $dataset"
    # python3 data_process/process_answer.py --input_file /mount/studenten-temp1/users/linchi/2024WS-FM/LLaVA-Med/data/${dataset}/test-answer-file.jsonl --output_file output/llava-med/answer/test-${dataset}_answer-file.jsonl
    echo "Processing gt"
    # python3 data_process/process_gt.py --input_file /mount/studenten-temp1/users/linchi/2024WS-FM/LLaVA-Med/data/${dataset}/test_question_answer_gt.jsonl --output_file output/llava-med/test_${dataset}_question_answer_gt.jsonl
    
    echo "Evaluating $dataset"
    python3 evaluation/eval.py --gt="output/llava-med/test_${dataset}_question_answer_gt.jsonl" --pred="output/llava-med/answer/test-${dataset}_answer-file.jsonl"
    echo "Done"
done

# # evaluate the model
# for dataset in "VQA-RAD" "PathVQA" "SLAKE"
# do
#     # python3 data_process/process_gt.py --input_file data/$dataset/test_question_answer_gt.jsonl --output_file data/$dataset/test_question_answer_gt.jsonl
#     echo "Evaluating $dataset"
#     python3 evaluation/eval.py --gt="data/$dataset/test_question_answer_gt.jsonl" --pred="output/vilt/vilt_test_${dataset}_answer-file.jsonl"
# done

# python3 evaluation/eval.py --gt=output/llava/test_question_answer_gt.jsonl --pred=output/llava/test_vqa_rad_answer-file.jsonl

# TEST: VQA-RAD__TINY-LLAVA
#python3 evaluation/eval.py --gt=data/VQA-RAD/test_question_answer_gt.jsonl --pred=OUTPUTS_jsonl/tinyllava_test_vqa_rad_answer_pred.jsonl

# TEST: SLAKE__TINY-LLAVA
#python3 evaluation/eval.py --gt=data/SLAKE/test_question_answer_gt.jsonl --pred=OUTPUTS_jsonl/tinyllava_test_slake_answer_pred.jsonl

# TEST: PathVQA__TINY-LLAVA
#python3 evaluation/eval.py --gt=data/PathVQA/test_question_answer_gt.jsonl --pred=OUTPUTS_jsonl/tinyllava_test_pathvqa_answer_pred.jsonl