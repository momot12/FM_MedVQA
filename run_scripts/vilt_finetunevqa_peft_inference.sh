

DATASET_NAME="VQA-RAD"

ckpt1="output/vilt-peft-finetune/ckpt/${DATASET_NAME}/bs_128_ep_1"
ckpt2="output/vilt-peft-finetune/ckpt/${DATASET_NAME}/bs_128_ep_10"
ckpt3="output/vilt-peft-finetune/ckpt/${DATASET_NAME}/bs_128_ep_100"
ckpt4="output/vilt-peft-finetune/ckpt/${DATASET_NAME}/bs_128_ep_1000"

array1=($ckpt1 $ckpt2 $ckpt3 $ckpt4)
array2=(1 10 100 1000)

# Get the length of the arrays
length=${#array1[@]}

# Iterate over the indices
for ((i=0; i<length; i++)); do
    
  python3 vilt_finetunevqa_peft_inference.py \
    --dataset_name ${DATASET_NAME} \
    --model_dir "${array1[i]}" \
    --epoch "${array2[i]}"

  python3 evaluation/eval.py \
    --gt=data/${DATASET_NAME}/test_question_answer_gt.jsonl \
    --pred=output/vilt_finetunevqa_test_${DATASET_NAME}_answer-file_epoch=${array2[i]}.jsonl
done



# python3 vilt_finetunevqa_peft_inference.py --dataset_name VQA-RAD --model_dir "output/vilt-peft-finetune/ckpt/2025-01-08-10:37 epoch=1000" --epoch 1000
# python3 vilt_finetunevqa_peft_inference.py --dataset_name SLAKE --model_dir "output/vilt-peft-finetune/ckpt/2025-01-08-17:55/bs_128_ep_1000"
