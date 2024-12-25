import torch
import os
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from datasets import Dataset
import argparse
import json
from peft import get_peft_model, LoraConfig
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="Models/llama2-7b-chat-hf", type=str)
parser.add_argument("--device", default="0", type=str)
args = parser.parse_args()
model_name = args.model_name
parser.add_argument("--data_path", default="./ConFiQA/data_train.json", type=str)
parser.add_argument("--points_path", default="./train/%s/check_points" % model_name[7:])
parser.add_argument("--save_path", default="./train/%s/save_model" % model_name[7:])
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
os.environ["WANDB_MODE"] = "disabled"

def create_path(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder {folder_path} Created")
    else:
        print(f"Folder {folder_path} Existed")


def load_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def qa_to_prompt(query, context):
    prompt = '{}\nQ: {}\nA: '.format(context, query)
    return prompt

def data_process(data):
    
    processed_data = []
    for d in data:
        orig_answer = d['orig_answer']
        cf_answer = d['cf_answer']
        prompt = qa_to_prompt(d['question'], d['cf_context'])
        chosen = "%s So the final answer is %s." % (d['cf_context_piece'], cf_answer)
        reject = "%s So the final answer is %s." % (d['orig_context_piece'], orig_answer)
        
        processed_data.append({"prompt": prompt, "chosen": chosen, "rejected": reject})
    return processed_data

def train_model(data, model_name):

    train_dataset = Dataset.from_list(data)
    
    load_kwargs = {
    "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
}
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model.gradient_checkpointing_enable()

    # 定义 PEFT 配置
    peft_config = LoraConfig(
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0,  # 设置为0以优化性能
        bias="none",  # 设置为"none"以优化性能
    )
    
    # Do model patching and add fast LoRA weights
    model = get_peft_model(model, peft_config)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=DPOConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            warmup_ratio=0.1,
            num_train_epochs=3,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            save_steps=400,
            # optim="adamw_8bit",
            seed=42,
            output_dir=args.points_path
        ),
        beta=0.1,
        train_dataset=train_dataset,
        # eval_dataset = YOUR_DATASET_HERE,
        processing_class=tokenizer,
        max_length=1024,
        max_prompt_length=512,
    )
    dpo_trainer.train()
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)

if __name__ == '__main__':
    
    create_path(args.save_path[:-11])
    create_path(args.points_path)
    create_path(args.save_path)

    data = load_json(args.data_path)
    processed_data = data_process(data)
    train_model(processed_data, args.model_name)
    
