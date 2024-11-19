# FM_MedVQA

### Tentative Plan: (https://github.com/jinlHe/PeFoMed?tab=readme-ov-file)  

Datasets: VQA-RAD, SLAKE  
1. Reproduce PeFoMed (MiniGPT-v2: a unified interface for completing many vision-language tasks including image description, visual question answering, and visual grounding, among others) 

2. Fine-tune reproduction but with less datasets
   --> Expecting performance to drop

4. Change pretrained LLM with(LLaVa-med, BioVilt, BLIP-2), fine-tune adapters (LoRA) with the new pretrained model
   --> Is it better or same as step 2 with less datasets

```
.
- pretrain.py
   - train.py
- finetune.py
   - test.py

./run_scripts
- bash run_pretrain.sh --model vilt_peft --datasets roco
   - pretrain.py ...
- bash run_finetune.sh --model vilt_peft --datasets roco
   - finetune.py
- 

./datasets/
- roco.py
- other_datasets.py

./models
- vilt.py
- llava.py
- vilt_peft.py
- llava_peft.py

./eval
- utils.py

```
