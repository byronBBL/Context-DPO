import json
from collections import defaultdict
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import random
from tqdm import tqdm
import ast
import torch
from transformers import AutoTokenizer, AutoModel, StoppingCriteria, StoppingCriteriaList, AutoModelForCausalLM

import argparse
from texttable import Texttable
import json
import random
import transformers
from tqdm import tqdm
import argparse
import pandas as pd
from datetime import datetime

import time
import re

model_name = 'model_name'


class LLamaQaStoppingCriteria(StoppingCriteria):
    def __init__(self, list_token_ids_sequence: list = []):
        self.token_ids_sequences = []
        self.lengths = []
        for token_ids_sequence in list_token_ids_sequence:
            self.token_ids_sequences.append(torch.tensor(token_ids_sequence, dtype=torch.long))
            self.lengths.append(len(token_ids_sequence))
        
    # @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # check the final {self.length} tokens
        stop = False
        for token_ids_sequence, length in zip(self.token_ids_sequences, self.lengths):
            if input_ids.shape[-1] < length:
                continue
            else:
                if bool(torch.all(input_ids[0, -length:] == token_ids_sequence.to(input_ids.device))):
                    stop = True
                    break
        return stop

def set_stop_words(tokenizer, stop):
    stop_words = stop
    list_stop_word_ids = []
    for stop_word in stop_words:
            stop_word_ids = tokenizer.encode('\n' + stop_word)[3:]
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
    stopping_criteria = StoppingCriteriaList()
    stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))
    return stopping_criteria
            
def call_llama(model, tokenizer, prompt, stopping_criteria, stop):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    sequences = model.generate(input_ids.cuda(), stopping_criteria = stopping_criteria, max_new_tokens = 512)[0, input_ids.shape[-1]:]
    decoded = tokenizer.decode(sequences, skip_special_tokens=True)
    for stop_word in stop:
        length_to_remove = len(stop_word)
        if decoded[-length_to_remove:] == stop_word:
            decoded = decoded[:-length_to_remove]
    output_str = decoded.strip()
    return output_str

def load_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# def call_gpt(cur_prompt, stop):
#     completion = client.chat.completions.create(
#     messages=[
#     {"role": "system", "content": "You are a helpful assistant. Please output the counter fact to the given real fact according to the following instruction and example in aligned format"},
#     {"role": "user", "content": cur_prompt}
#   ],
#     model= model_name,
#     stop = stop,
#     temperature=0,
#     max_tokens=64,
# )

#     message = completion.choices[0].message
#     content = message.content
#     return content

def convert_str_to_list(data):
    if isinstance(data, dict):
        return {key: convert_str_to_list(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_str_to_list(element) for element in data]
    elif isinstance(data, str):
        # 尝试将字符串解析为 Python 表达式
        try:
            # 检查字符串是否是列表格式
            result = ast.literal_eval(data)
            if isinstance(result, list):
                return result
        except (ValueError, SyntaxError):
            pass
    return data

def generate_cloze(subject, property_name, target, rela_templates):
    # 如果 property_name 存在于模板中，则用模板格式化
    if property_name in rela_templates:
        template = rela_templates[property_name]
        # 使用模板生成事实断言，替换 [X] 和 [Y]
        cloze = template.replace("[subject]", subject).replace("[target]", target)
        return cloze
    else:
        # 如果没有模板，使用默认格式
        return f"{subject} has {property_name} as {target}"
    
qustion_prompt_qa = '''Given a triple in the form of `(subject, relation, target)`, generate a question that uses the `subject` and `relation` to ask about the `target`. The question should be natural, concise, and directly address the relationship between the `subject` and `target`.
Triple: (United States, capital, Washington, D.C.)  
Question: What is the capital of the United States?

Triple: (United States, head of government, Joe Biden)  
Question: Who is the current head of the United States government?

Triple: (United States, official language, English)  
Question: What is the official language of the United States?

Triple: ({subject}, {relation}, {target})  
Question: '''
    
def generate_question(subject, relation, target, model, tokenizer, stopping_criteria, stop):
    
    prompt = qustion_prompt_qa.format(subject=subject, relation=relation, target=target)
    gen = call_llama(model, tokenizer, prompt, stopping_criteria, stop)
    return gen

qustion_prompt_multihop = "You are a sophisticated {hop_num}-hop question generator. Given a chain of Wikidata triples, generate a question that asks about the final tail entity ({tail}) in the chain using only the starting head entity ({head}). Do not include any bridge entities in the question; instead, phrase the question as if directly asking about the relationship from the head entity to the tail entity."

hop_example = [
    '''Input: ('Jacques Necker', 'employer', 'University of Geneva'), ('University of Geneva', 'headquarters location', 'Geneva')\nOutput: What is the location of the headquarters of the employer of Jacques Necker?\n\nInput: ('Percival Lowell', 'educated at', 'Harvard University'), ('Harvard University', 'headquarters location', 'Cambridge')\nOutput: In which city is the institution located where Percival Lowell received his education?\n\nInput: ('Gordon Moore', 'country of citizenship', 'United States of America'), ('United States of America', 'capital', 'Washington, D.C.')\nOutput: What is the capital of the country where Gordon Moore holds citizenship?\n\nInput: {triples}\nOutput: '''
    ,
    '''Input: ('Kim Kardashian', 'spouse', 'Kanye West'), ('Kanye West', 'genre', 'hip hop music'), ('hip hop music', 'country of origin', 'United States of America')\nOutput: What is the country of origin of the genre associated with the spouse of Kim Kardashian?\n\nInput: ('Nicholas of Tolentino', 'religion or worldview', 'Catholic Church'), ('Catholic Church', 'founded by', 'Jesus Christ'), ('Jesus Christ', 'place of birth', 'Bethlehem')\nOutput: In which city was the founder of the religion that Nicholas of Tolentino adhered to born?\n\nInput: ('Boston', 'head of government', 'Marty Walsh'), ('Marty Walsh', 'educated at', 'Boston College'), ('Boston College', 'headquarters location', 'Chestnut Hill')\nOutput: In what city is the headquarters of the institution where the head of government of Boston was educated located?\n\nInput: {triples}\nOutput: '''
    ,
    '''Input: Input: ('Xbox Live', 'developer', 'Microsoft'), ('Microsoft', 'chief executive officer', 'Satya Nadella'), ('Satya Nadella', 'place of birth', 'Hyderabad'), ('Hyderabad', 'continent', 'Asia')\nOutput: Which continent is home to the birthplace of the CEO of Xbox Live developer?\n\nInput: ('Winnie the Pooh', 'creator', 'A. A. Milne'), ('A. A. Milne', 'child', 'Christopher Robin Milne'), ('Christopher Robin Milne', 'country of citizenship', 'United Kingdom'), ('United Kingdom', 'official language', 'English')\nOutput: What is the official language of the country where the child of Winnie the Pooh’s creator holds citizenship?\n\nInput: ('watchOS', 'developer', 'Apple Inc.'), ('Apple Inc.', 'chief executive officer', 'Tim Cook'), ('Tim Cook', 'country of citizenship', 'United States of America'), ('United States of America', 'capital', 'Washington, D.C.')\nOutput: What is the capital of the country where the CEO of the developer of watchOS holds citizenship?\n\nInput: {triples}\nOutput: '''
]


def generate_question_multihop(triples, hop_num, head, tail, model, tokenizer, stopping_criteria, stop):
    
    prompt = qustion_prompt_multihop.format(hop_num=hop_num+2, head=head, tail=tail) + ' ' + hop_example[hop_num].format(triples=triples)
    gen = call_llama(model, tokenizer, prompt, stopping_criteria, stop)
    return gen
    
context_prompt_qa = '''Considering {facts}, generate a brief description of the entity: {entity}, approximately 100 words long. Ensure that {ans} is accurately mentioned in the description, and make sure that no other forms or variations of {ans} (such as different parts of speech) appear in the text.'''

def generate_llm_qa(entity, cloze, ans, ans_alias, model, tokenizer, stopping_criteria, stop):
   
    prompt = context_prompt_qa.format(entity=entity, facts=cloze, ans=ans)
    ans_alias.append(ans)
   
    gen = call_llama(model, tokenizer, prompt, stopping_criteria, stop)
    cnt = 1
   
    while cnt <= 3 and not any(re.search(r'\b' + re.escape(alias) + r'\b', gen) for alias in ans_alias):
        gen = call_llama(model, tokenizer, prompt, stopping_criteria, stop)
        cnt += 1
   
    if not any(re.search(r'\b' + re.escape(alias) + r'\b', gen) for alias in ans_alias):
        return ""
   
    return gen
    
def replace_cf(context, orig_alias, cf, orig_adj, cf_adj):
    # 遍历 orig_alias 中的每个实体别名
    for orig in orig_alias:
        # 使用正则表达式匹配大小写不敏感的原始实体
        # 使用\b表示单词边界，确保我们替换的是完整的单词，而不是单词的一部分
        pattern = r'\b' + re.escape(orig) + r'\b'
        
        # 替换文本中的原实体为替换实体
        context = re.sub(pattern, cf, context, flags=re.IGNORECASE)
    
    pattern = r'\b' + re.escape(orig_adj) + r'\b'
    context = re.sub(pattern, cf_adj, context, flags=re.IGNORECASE)
    
    return context

def generate_context_multihop_mc(triple_data, rele_template, label_alias, entity_adj, model, tokenizer, stopping_criteria, stop):
    
    context_data = []
    
    for data in tqdm(triple_data):

        orig_path = convert_str_to_list(data['orig_path'])
        cf_path = convert_str_to_list(data['cf_path'])
        cf_orig_path = convert_str_to_list(data['orig'])
        data['orig_context_piece'] = []
        data['cf_context_piece'] = []
        data['cf_context'] = []
        data['orig_context'] = []
        is_gen = True
        
        for orig, cf, cf_orig in zip(orig_path, cf_path, cf_orig_path):
            fact = generate_cloze(label_alias[orig[0]]['label'], orig[1], label_alias[orig[2]]['label'], rele_template)
            counter_fact = generate_cloze(label_alias[cf[0]]['label'], cf[1], label_alias[cf[2]]['label'], rele_template)
            counter_fact_orig = generate_cloze(label_alias[cf_orig[0]]['label'], cf_orig[1], label_alias[cf_orig[2]]['label'], rele_template)
            data['orig_context_piece'].append(fact)
            data['cf_context_piece'].append(counter_fact)
            gen1 = generate_llm_qa(label_alias[orig[0]]['label'], fact, label_alias[orig[2]]['label'], label_alias[orig[2]]['alias'], model, tokenizer, stopping_criteria, stop)
            if gen1 == "":
                is_gen = False
                break
            gen2 = generate_llm_qa(label_alias[cf_orig[0]]['label'], counter_fact_orig, label_alias[cf_orig[2]]['label'], label_alias[cf_orig[2]]['alias'], model, tokenizer, stopping_criteria, stop)
            if gen2 == "":
                is_gen = False
                break
            
            cf_gen2 = replace_cf(gen2, [label_alias[cf_orig[2]]['label']] + label_alias[cf_orig[2]]['alias'], label_alias[cf[2]]['label'], entity_adj[cf_orig[2]]['adj'], entity_adj[cf[2]]['adj'])
            data['orig_context'].append(gen1)
            data['cf_context'].append(cf_gen2)
            
        if is_gen:
            data['orig_context'] = ' '.join(data['orig_context'])
            data['cf_context'] = ' '.join(data['cf_context'])
            data['orig_context_piece'] = '. '.join(data['orig_context_piece']) + '.'
            data['cf_context_piece'] = '. '.join(data['cf_context_piece']) + '.'
            context_data.append(data)

    return context_data

def generate_context_multihop_mr(triple_data, rele_template, label_alias, entity_adj, model, tokenizer, stopping_criteria, stop):
    
    context_data = []
    cnt = 0
    
    for data in tqdm(triple_data):

        orig_path = convert_str_to_list(data['orig_path'])
        cf_path = convert_str_to_list(data['cf_path'])
        data['orig_context_piece'] = []
        data['cf_context_piece'] = []
        data['cf_context'] = []
        data['orig_context'] = []
        is_gen = True
        is_cf = False
        
        for orig, cf in zip(orig_path, cf_path):
            
            fact = generate_cloze(label_alias[orig[0]]['label'], orig[1], label_alias[orig[2]]['label'], rele_template)
            counter_fact = generate_cloze(label_alias[cf[0]]['label'], cf[1], label_alias[cf[2]]['label'], rele_template)
            data['orig_context_piece'].append(fact)
            data['cf_context_piece'].append(counter_fact)
            gen1 = generate_llm_qa(label_alias[orig[0]]['label'], fact, label_alias[orig[2]]['label'], label_alias[orig[2]]['alias'], model, tokenizer, stopping_criteria, stop)
            if gen1 == "": 
                is_gen = False
                break
            if is_cf:
                gen2 = generate_llm_qa(label_alias[cf[0]]['label'], counter_fact, label_alias[cf[2]]['label'], label_alias[cf[2]]['alias'], model, tokenizer, stopping_criteria, stop)
                if gen2 == "": 
                    is_gen = False
                    break
                data['orig_context'].append(gen1)
                data['cf_context'].append(gen2)
            else:
                if str(cf) == data['cf_triple']:
                    is_cf = True
                    cf_gen1 = replace_cf(gen1, [label_alias[orig[2]]['label']] + label_alias[orig[2]]['alias'], label_alias[cf[2]]['label'], entity_adj[orig[2]]['adj'], entity_adj[cf[2]]['adj'])
                    data['orig_context'].append(gen1)
                    data['cf_context'].append(cf_gen1)
                else:
                    data['orig_context'].append(gen1)
                    data['cf_context'].append(gen1)
        if is_gen:
            data['orig_context'] = ' '.join(data['orig_context'])
            data['cf_context'] = ' '.join(data['cf_context'])
            data['orig_context_piece'] = '. '.join(data['orig_context_piece']) + '.'
            data['cf_context_piece'] = '. '.join(data['cf_context_piece']) + '.'
            context_data.append(data)
        

    return context_data

def generate_context_qa(triple_data, rele_template, label_alias, entity_adj, model, tokenizer, stopping_criteria, stop):
    
    context_data = []
    cnt = 0
    
    for data in tqdm(triple_data):
        orig = convert_str_to_list(data['orig_path'])[0]
        cf = convert_str_to_list(data['cf_path'])[0]

        fact = generate_cloze(label_alias[orig[0]]['label'], orig[1], label_alias[orig[2]]['label'], rele_template)
        counter_fact = generate_cloze(label_alias[cf[0]]['label'], cf[1], label_alias[cf[2]]['label'], rele_template)
        data['orig_context_piece'] = fact + '.'
        data['cf_context_piece'] = counter_fact + '.'
        
        gen = generate_llm_qa(label_alias[orig[0]]['label'], fact, label_alias[orig[2]]['label'] , label_alias[orig[2]]['alias'], model, tokenizer, stopping_criteria, stop)
        cf_gen = replace_cf(gen, [label_alias[orig[2]]['label']] + label_alias[orig[2]]['alias'], label_alias[cf[2]]['label'], entity_adj[orig[2]]['adj'], entity_adj[cf[2]]['adj'])
        if gen == "": continue
        
        context_data.append(data)
        context_data[-1]['orig_context'] = gen
        context_data[-1]['cf_context'] = cf_gen
        

    return context_data

if __name__ == "__main__":
    
    output_file = 'facts_filtered.json'  # 输出的 triples 文件
    error_log_file = 'facts_error_log.txt'  # 错误日志文件
    
    data_qa = load_json('./data_preprocess/data/context_data_qa_ques.json')
    data_mr = load_json('./data_preprocess/data/context_data_mr_ques.json')
    data_mc = load_json('./data_preprocess/data/context_data_mc_ques.json')
    label_alias = load_json('./data_preprocess/entity_lable_alias.json')
    rele_template = load_json("./rela_template.json")
    entity_adj = load_json('./data_preprocess/entity_adj.json')
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", low_cpu_mem_usage = True, torch_dtype=torch.float16, trust_remote_code=True)
    model.cuda()
    stop = ['\n\nTriple:', '\n\nInput:']
    stopping_criteria = set_stop_words(tokenizer, stop)

    for e_id, e in label_alias.items():
        label_alias[e_id]['alias'] = sorted(label_alias[e_id]['alias'], key=len, reverse=True)
    
    context_data_qa = generate_context_qa(data_qa, rele_template, label_alias, entity_adj, model, tokenizer, stopping_criteria, stop)
    context_data_mr = generate_context_multihop_mr(data_mr, rele_template, label_alias, entity_adj, model, tokenizer, stopping_criteria, stop)
    context_data_mc = generate_context_multihop_mc(data_mc, rele_template, label_alias, entity_adj, model, tokenizer, stopping_criteria, stop)


    # 保存过滤后的 triples 到新的文件中
    save_json(context_data_qa, './data_preprocess/data/context_data_qa.json')
    save_json(context_data_mr, './data_preprocess/data/context_data_mr.json')
    save_json(context_data_mc, './data_preprocess/data/context_data_mc.json')

