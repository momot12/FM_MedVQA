# A comparative study of Medical VQA models

#### RQ: Can a smaller-scale visual language model achieve comparable performance in medical visual question answering tasks?

#### Contribution summary:

1. **Baseline Performance of Pre-trained Models**  
   We provide an overview of the current state of performance of pre-trained models in general Visual Question Answering (VQA), such as **[ViLT](https://proceedings.mlr.press/v139/kim21k/kim21k.pdf)** and **[TinyLLaVA](https://github.com/TinyLLaVA/TinyLLaVA_Factory/tree/tinyllava_bench)**, as baselines.

2. **Pre-training and Fine-tuning with Connecting Layers**  
   We focus on pre-training and fine-tuning only a connecting layer, such as **LoRA** [(Hu et al., 2021)](https://arxiv.org/abs/2106.09685).

3. **Comparison with SOTA Medical VQA Models**  
   We compare the performance of current state-of-the-art (SOTA) medical pre-trained VQA models, such as **[LLaVA-Med](https://github.com/microsoft/LLaVA-Med)**, against models described in (2).

4. **Quantitative and Qualitative Analysis**  
   We provide both quantitative and qualitative analyses of the generated answers. Additionally, we incorporate a semantic metric (**[F1_BERT](https://huggingface.co/spaces/evaluate-metric/bertscore)**) for evaluation.



## Overview

| Phases | Model | (1) Domain adaptation | (2) Inference / Finetune |
| --- | --- | --- | --- |
| A | ViLT | - | VQA-RAD, SLAKE, PathVQA (Inference) |
| A | TinyLLaVa | - | VQA-RAD, SLAKE, PathVQA (Inference) |
| B | ViLT + Adapter | ROCOv2 | VQA-RAD, SLAKE, PathVQA (Finetune) |
| B | TinyLLaVA + Connector | ROCOv2 | VQA-RAD, SLAKE, PathVQA (Finetune) |
| C | LLaVA-Med | - | VQA-RAD, SLAKE, PathVQA (Inference) |


### Models
| Model | Vision | LLM | Connector/Adapter | Domain | Model Size |
| --- | --- | --- | --- | --- | --- |
| ViLT | Linear Projection | Transformer | + LoRA (Adapter) | General | 87.4 M |
| TinyLLaVA | SigLIP | Phi-2 | + 2-layer MLP <br>+ GELU <br>(Connector) | General | 3.1 B |
| LLaVA-Med | LLaVA | Mistral-7B | Linear Projection | Medical | 7.0 B |

### Datasets
| Dataset | Stage | # Images | # QA Pair | Image Type | Text |
| --- | --- | --- | --- | --- | --- |
| [ROCOv2](https://huggingface.co/datasets/eltorio/ROCOv2-radiology) | Pre-train | 79,789 | - | Radiology | Caption |
| [VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) | Fine-tune | 315 | 3,515 | Radiology | Open/close-ended questions MCQ |
| [SLAKE (ENG)](https://huggingface.co/datasets/mdwiratathya/SLAKE-vqa-english) | Fine-tune | 642 | 14,000 | Radiology | Open/close-ended questions MCQ |
| [PathVQA](https://huggingface.co/datasets/flaviagiammarino/path-vqa) | Fine-tune | 4,998 | 32,799 | Pathology | Open/close-ended questions MCQ |

## Evaluation
### Quantitative Metrics:
Implementation is taken from [BESTMVQA](https://github.com/emmali808/BESTMVQA/blob/master/LLaVA-Med/llava/eval/run_eval.py#L103).  
How to run (choose corresponding gt/pred files): ```python3 eval.py --gt=gt_file.json --pred=prediction_file.json```

### Qualitative Metric:
This is a customize evaluation meant to focus on the semantic aspects of the generated sentences.  
The implementation can be found here: [qualitative_custom_eval.py](EVALUATION/qualitative_custom_eval.py).

Mainly three cases are distinguished:  
#### Case 1: Closed Questions
- For **closed questions**, if the prediction starts with "yes" or "no", it is truncated to **yes/no**.
- For **match** and **mismatch**, the **F1-score** is set to:
  - **1.0** for a match.
  - **0.0** for a mismatch.
#### Case 2: Open Questions
- For **open questions**, the evaluation is based on substring matching:
  - If the **ground truth string** is found as a substring in the prediction, we compute: F1_BERT(original ground truth, substring of prediction)
  - If **no substring matching** occurs, we compute: F1_BERT(original ground truth, original prediction)
#### Case 3: Special Cases
- For cases where the **ground truth** is a closed answer, but the prediction is an open text:
  - The **F1BERT Score** is computed for the original strings.


## Acknowledgements & Script References
- Architecture inspiration: [PeFoMed](https://github.com/jinlHe/PeFoMed)   
- Quantitative metrics: [BESTMVQA](https://github.com/emmali808/BESTMVQA/blob/master/LLaVA-Med/llava/eval/run_eval.py#L103)
- Inference (LLaVA-Med): [Multimodal Chat Inference](https://github.com/microsoft/LLaVA-Med?tab=readme-ov-file#4-multimodal-chat-inference)
- Pretrain/Finetune scripts (TinyLLaVA): [MODELS/tinyllava/README.md](MODELS/tinyllava/README.md)
- Pretrain/Finetune scripts (ViLT): [MODELS/vilt/README.md](MODELS/vilt/README.md)

## Authors
For questions, please contact one of the authors:
- Cheng-Wei Lin (st191423@stud.uni-stuttgart.de)
- Momo Takamatsu (st172293@stud.uni-stuttgart.de)
- Muhammad Gema Akbar (st191386@stud.uni-stuttgart.de)

<br><br>
Last updated on  27.01.2025
