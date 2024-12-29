Context-DPO: Aligning Language Models for Context-Faithfulness
===

[![Arxiv](https://img.shields.io/badge/arXiv-2412.15280-B21A1B)](https://arxiv.org/abs/2412.15280)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-blue)](https://github.com/huggingface/transformers)

Code and data for the paper "Context-DPO: Aligning Language Models for Context-Faithfulness"

Paper: https://arxiv.org/abs/2412.15280  
Authors: [Baolong Bi](https://byronbbl.github.io/) $^{1}$, Shaohan Huang $^{2}$, Yiwei Wang $^{3}$, Tianchi Yang $^{2}$, Zihan Zhang $^{2}$, Haizhen Huang $^{2}$, Lingrui Mei $^{1}$, Junfeng Fang $^{4}$, Zehao Li $^{1}$, Furu Wei $^{2}$, Weiwei Deng $^{2}$, Feng Sun $^{2}$, Qi Zhang $^{2}$, [Shenghua Liu](https://shenghua-liu.github.io/)$^{1}$  

$^1$ University of Chinese Academy of Sciences, $^2$ Microsoft Corporation, $^3$ University of California, Merced, $^4$ National University of Singapore  

## Overview

![Context-DPO](overview.jpg)

Reliable responses from large language models (LLMs) require adherence to user instructions and retrieved information. While alignment techniques help LLMs align with human intentions and values, improving context-faithfulness through alignment remains underexplored. To address this, we propose \textbf{Context-DPO}, the first alignment method specifically designed to enhance LLMs' context-faithfulness. We introduce \textbf{ConFiQA}, a benchmark that simulates Retrieval-Augmented Generation (RAG) scenarios with knowledge conflicts to evaluate context-faithfulness. By leveraging faithful and stubborn responses to questions with provided context from ConFiQA, our Context-DPO aligns LLMs through direct preference optimization. Extensive experiments demonstrate that our Context-DPO significantly improves context-faithfulness, achieving 35\% to 280\% improvements on popular open-source models. Further analysis demonstrates that Context-DPO preserves LLMs' generative capabilities while providing interpretable insights into context utilization.

## Context-Faithful Models
The four aligned models for Context-Faithfulness are now available on huggingface-hub:

| Model Name                | HF Checkpoint                                                                              | License |
|:--------------------------|:-------------------------------------------------------------------------------------------| :--- |
| Context-Faithful-LLaMA-2-7b-chat-hf | 🤗 [Bibaolong/Context-Faithful-LLaMA-2-7b-chat-hf](https://huggingface.co/Bibaolong/Context-Faithful-LLaMA-2-7b-chat-hf) | [Llama2-Chat](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)|
| Context-Faithful-LLaMA-3-8b-instruct  | 🤗 [Bibaolong/Context-Faithful-LLaMA-3-8b-instruct](https://huggingface.co/Bibaolong/Context-Faithful-LLaMA-3-8b-instruct)         | [Llama3-Instruct](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)|
| Context-Faithful-Mistral-7B-instruct  | 🤗 [Bibaolong/Context-Faithful-Mistral-7B-instruct-v0.2](https://huggingface.co/Bibaolong/Context-Faithful-Mistral-7B-instruct-v0.2)         | [Mistral-Instruct](https://mistral.ai/contact/)|
| Context-Faithful-Qwen2-7B-Instruct  | 🤗 [Bibaolong/Context-Faithful-Qwen2-7B-Instruct](https://huggingface.co/Bibaolong/Context-Faithful-Qwen2-7B-Instruct)         | [Qwen-Instruct](https://github.com/QwenLM/Qwen/blob/main/LICENSE)|

## Setup

Before you begin, make sure to install the necessary packages: `torch`, `transformers`, `trl`, `peft`, `datasets`, `tqdm` and so on. You can directly run the following command: `pip install -r requirements.txt` to download the corresponding versions.

## Datasets

In this work, we introduce the **ConFiQA** benchmark to evaluate the context-faithfulness of LLMs in real-world RAG scenarios involving knowledge conflicts.
ConFiQA consists of three datasets: **QA** (Question-Answering), **MR** (Multi-hop Reasoning), and **MC** (Multi-Conflicts).
QA features single-hop question-answering tasks with context containing one corresponding counterfactual, while MR and MC involve multi-hop reasoning tasks with context containing one and multiple related counterfactuals, respectively.

**ConFiQA** is available in this repository. You can also download the **Natural Questions with knowledge conflict** dataset from [Google Drive](https://drive.google.com/file/d/1DJ1ajmLNAKVTBWnM7SkP93EYQ2cav3Mk/view).  To perform context-faithfulness evaluation on NQ dataset, follow the instructions provided in [GitHub Repository](https://github.com/wzhouad/context-faithful-llm/tree/main?tab=readme-ov-file).  



## Experiments

Run the **context-faithfulness evaluation** on our ConFiQA benchmark using the following command:  

```bash
python evaluation.py --model_name ./model_path --data_path ./ConFiQA/ConFiQA-QA.json --output_path ./result/output.json
```

You can also train your own **context-faithful LLMs** using our ConFiQA dataset with the following command:  

```bash
python train_dpo.py --model_name ./model_path --data_path ./ConFiQA/data_train.json --points_path ./check_points --save_path ./context-faithful_model
```
We recommend selecting a proportion of data from the QA, MR, and MC tasks in ConFiQA to serve as training data in `./ConFiQA/data_train.json` for DPO training.


## Citation
If you find our work useful, please cite our paper:
```bibtex
@article{bi2024contextdpo,
  title={Context-DPO: Aligning Large Language Models for Context-Faithfulness},
  author={Bi, Baolong and Huang, Shaohan and Wang, Yiwei and Yang, Tianchi and Zhang, Zihan and Huang, Haizhen and Mei, Lingrui and Fang, Junfeng and Li, Zehao and Wei, Furu and Deng, Weiwei and Sun, Feng and Zhang, Qi and Liu, Shenghua},
  journal={arXiv preprint arXiv:2412.15280},
  year={2024},
  url={https://arxiv.org/abs/2412.15280}
}
```
