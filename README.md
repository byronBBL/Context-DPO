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
The four aligned models for Context-Faithfulness are now available on huggingface-hub: [Context-Faithful LLMs Collections](https://huggingface.co/collections/Bibaolong/context-faithful-llms-676b783a6f93172ba99751cf)

| Model Name                | HF Checkpoint                                                                              | License |
|:--------------------------|:-------------------------------------------------------------------------------------------| :--- |
| Context-Faithful-LLaMA-2-7b-chat-hf | 🤗 [Bibaolong/Context-Faithful-LLaMA-2-7b-chat-hf](https://huggingface.co/Bibaolong/Context-Faithful-LLaMA-2-7b-chat-hf) | [Llama2-Chat](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)|
| Context-Faithful-LLaMA-3-8b-instruct  | 🤗 [Bibaolong/Context-Faithful-LLaMA-3-8b-instruct](https://huggingface.co/Bibaolong/Context-Faithful-LLaMA-3-8b-instruct)         | [Llama3-Instruct](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)|
| Context-Faithful-Mistral-7B-instruct  | 🤗 [Bibaolong/Context-Faithful-Mistral-7B-instruct-v0.2](https://huggingface.co/Bibaolong/Context-Faithful-Mistral-7B-instruct-v0.2)         | [Mistral-Instruct](https://mistral.ai/contact/)|
| Context-Faithful-Qwen2-7B-Instruct  | 🤗 [Bibaolong/Context-Faithful-Qwen2-7B-Instruct](https://huggingface.co/Bibaolong/Context-Faithful-Qwen2-7B-Instruct)         | [Qwen-Instruct](https://github.com/QwenLM/Qwen/blob/main/LICENSE)|

### Usage Instructions

1. **Download the model and adapter files**:  
   Make sure you have both the pre-trained model and the adapter files.

2. **Update the adapter configuration**:  
   - Open the `adapter_config.json` file.  
   - Locate the field `"base_model_name_or_path": "$$"`.  
   - Replace `$$` with the path or name of your local model.

3. **Load the model**:  
   After updating the configuration, you can load the model and the adapter for use.

4. **Run inference**:  
   Once the model and adapter are loaded, you can use the model to perform tasks as needed.

## Setup

Before you begin, make sure to install the necessary packages: `torch`, `transformers`, `trl`, `peft`, `datasets`, `tqdm` and so on. You can directly run the following command: `pip install -r requirements.txt` to download the corresponding versions.

## Datasets Overview

In this work, we introduce the **ConFiQA** benchmark to evaluate the context-faithfulness of LLMs in real-world RAG scenarios involving knowledge conflicts.
ConFiQA consists of three datasets: **QA** (Question-Answering), **MR** (Multi-hop Reasoning), and **MC** (Multi-Conflicts).
QA features single-hop question-answering tasks with context containing one corresponding counterfactual, while MR and MC involve multi-hop reasoning tasks with context containing one and multiple related counterfactuals, respectively.

**ConFiQA** is available in this repository. You can also download the **Natural Questions with knowledge conflict** dataset from [Google Drive](https://drive.google.com/file/d/1DJ1ajmLNAKVTBWnM7SkP93EYQ2cav3Mk/view).  To perform context-faithfulness evaluation on NQ dataset, follow the instructions provided in [GitHub Repository](https://github.com/wzhouad/context-faithful-llm/tree/main?tab=readme-ov-file).

## ConFiQA Format

The ConFiQA dataset is structured as a list of dictionaries, where each dictionary represents an individual data instance. Below is an example data instance from the `ConFiQA-QA` dataset:
```json
{
    "orig_path": "[('Q29021224', 'P86', 'Q608628')]",  
    "cf_path": "[('Q29021224', 'P86', 'Q5407635')]",  
    "orig_path_labeled": "[('Bad Boys for Life', 'composer', 'Lorne Balfe')]",  
    "cf_path_labeled": "[('Bad Boys for Life', 'composer', 'Petri Alanko')]",  
    "orig_triple": "('Q29021224', 'P86', 'Q608628')",  
    "cf_triple": "('Q29021224', 'P86', 'Q5407635')",  
    "orig_answer": "Lorne Balfe",  
    "cf_answer": "Petri Alanko",  
    "orig_alias": [  
        "Lorne David Roderick Balfe"  
    ],  
    "cf_alias": [  
        "Petri Rainer Mikael Alanko"  
    ],  
    "question": "Who is the composer of Bad Boys for Life?",  
    "orig_context_piece": "Bad Boys for Life was composed by Lorne Balfe.",  
    "cf_context_piece": "Bad Boys for Life was composed by Petri Alanko.",  
    "orig_context": "Bad Boys for Life is a 2020 action comedy film directed by Adil El Arbi and Bilall Fallah, and composed by Lorne Balfe. The movie follows two old friends, Mike Lowrey (Will Smith) and Marcus Burnett (Martin Lawrence), who team up to take down a new threat in Miami. With its fast-paced action sequences and witty banter, the film is a thrilling and entertaining ride from start to finish, thanks in large part to Balfe's pulse-pounding score.",  
    "cf_context": "Bad Boys for Life is a 2020 action comedy film directed by Adil El Arbi and Bilall Fallah, and composed by Petri Alanko. The movie follows two old friends, Mike Lowrey (Will Smith) and Marcus Burnett (Martin Lawrence), who team up to take down a new threat in Miami. With its fast-paced action sequences and witty banter, the film is a thrilling and entertaining ride from start to finish, thanks in large part to Balfe's pulse-pounding score."  
}
```

- **`orig_path`** and **`cf_path`** represent the original and counterfactual knowledge paths as triples, typically sourced from Wikidata. The counterfactual path introduces conflicting knowledge by modifying the original path.  

- **`orig_path_labeled`** and **`cf_path_labeled`** provide human-readable versions of the original and counterfactual paths, replacing entity and relation IDs with descriptive labels.  

- **`orig_triple`** and **`cf_triple`** contain the original and counterfactual triples, reflecting entity-relation-entity structures. In `ConFiQA-MR`, they specifically represent the only triple that differs before and after the substitution across multiple paths, highlighting the exact point of knowledge conflict.   

- **`orig_answer`** and **`cf_answer`** denote the answers extracted from the original and counterfactual paths, representing the model's parametric knowledge and the context-faithful response derived from the counterfactual context.  **`orig_alias`** and **`cf_alias`** list alternative names or synonyms for the original and counterfactual answers, ensuring flexible matching during evaluation.  

- **`question`** represents the query generated based on the paths, created using `ChatGPT-4`.  

- **`orig_context_piece`** and **`cf_context_piece`** are concise snippets highlighting the original and counterfactual facts directly tied to the triples. These snippets can be utilized for end-to-end training.  

- **`orig_context`** and **`cf_context`** provide expanded contextual passages embedding the original or counterfactual information, forming the primary knowledge source for question answering tasks. `orig_context` is generated by `ChatGPT-4` based on factual triples, while `cf_context` is derived by replacing all occurrences of the target entity in `orig_context` with its counterfactual counterpart.


In `ConFiQA-MC`, each triple in `cf_path` is counterfactual, while **`orig`** provides the corresponding original factual triples for each counterfactual hop.



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

## Bugs or Qustions?

If you have any questions related to the repo or the paper, or you encounter any problems when using the datasets/code, feel free to email Baolong Bi (bibaolong23z@ict.ac.cn) or open an issue!

## Citation

Please cite our paper if it's helpful to your work!
```bibtex
@article{bi2024context,
  title={Context-DPO: Aligning Language Models for Context-Faithfulness},
  author={Bi, Baolong and Huang, Shaohan and Wang, Yiwei and Yang, Tianchi and Zhang, Zihan and Huang, Haizhen and Mei, Lingrui and Fang, Junfeng and Li, Zehao and Wei, Furu and others},
  journal={arXiv preprint arXiv:2412.15280},
  year={2024}
}
```
