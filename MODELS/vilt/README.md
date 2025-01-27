## ViLT scripts

**Direct inference** is done using this checkpoint:
- Checkpoint: [huggingface hub: vilt-b32-finetuned-vqa](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa) (already fine-tuned for VQA)
- Implementation script: [vilt_inference.py](MODELS/vilt/vilt_inference.py)


Further **pre-training** (domain adaptation) and fine-tuning is done taking the following pre-trained checkpoints:
- [Huggingface hub: vilt-b32-mlm](https://huggingface.co/dandelin/vilt-b32-mlm)
- Implementation: [vilt_peft_pretrain.py](MODELS/vilt/vilt_peft_pretrain.py)
- Further pre-training: on medical image caption pairs using image-text match loss and mask language modeling loss. 

For **fine-tuning** on medical VQA, we load the domain adapted checkpoint and fine-tune it with cross entropy loss. We follow the original paper, formulating it into a classification task. See [custom_vilt.py](medvqa/datasets/vilt/custom_vilt.py), [vilt/datasets.py](medvqa/datasets/vilt/datasets.py), and [vilt_peft_finetune_inference.py](MODELS/vilt/vilt_peft_finetune_inference.py) for implementation details.