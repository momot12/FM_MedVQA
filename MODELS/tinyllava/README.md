## TinyLLaVA scripts
The code refers to the original [TinyLLaVA_Factory Repo](https://github.com/TinyLLaVA/TinyLLaVA_Factory/tree/tinyllava_bench).

**Direct inference** is done using this checkpoint:
- Checkpoint: ⁠[TinyLLaVA-1.5B](https://huggingface.co/bczhou/TinyLLaVA-1.5B)
- Implementation script: [tiny_llava_inference.py](FM_MedVQA/MODELS/tinyllava/tiny_llava_inference.py)

Further **pre-training** (domain adaptation) and fine-tuning is done taking the following pre-trained checkpoints:
- ⁠Small-scale LLM: [huggingface hub: microsoft phi-2](https://huggingface.co/microsoft/phi-2)
- ⁠⁠Vision Tower: [huggingface hub: SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384)

We customize the datasets following the descriptions in [CUSTOM_FINETUNE.md](https://github.com/TinyLLaVA/TinyLLaVA_Factory/blob/main/CUSTOM_FINETUNE.md).   
The **pre-train and fine-tune** scripts are based on [pretrain.sh](https://github.com/TinyLLaVA/TinyLLaVA_Factory/blob/tinyllava_bench/scripts/tiny_llava/pretrain.sh) and [finetune.sh](https://github.com/TinyLLaVA/TinyLLaVA_Factory/blob/tinyllava_bench/scripts/tiny_llava/finetune.sh) and modified accordingly. See [tinyllava_finetuned_inference.py](FM_MedVQA/MODELS/tinyllava/tinyllava_finetuned_inference.py) for modification details.