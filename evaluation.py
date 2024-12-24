import re
import ast
import string
import json
import re
import argparse
from tqdm import tqdm
import os
import torch
from transformers import AutoTokenizer, AutoModel, StoppingCriteria, StoppingCriteriaList, AutoModelForCausalLM
import logging


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

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

negation_words = [
    "no", "not", "never", "none", "cannot", "nobody", "nothing", "nowhere", 
    "neither", "nor", "without", "hardly"
]

def exact_match_score(prediction, ground_truth, is_cf):
    contains_negation = any(word in prediction.split() for word in negation_words)
    return (not contains_negation if is_cf else True) and (normalize_answer(prediction) == normalize_answer(ground_truth))    

def recall_score(prediction, ground_truth, is_cf):
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    
    contains_negation = any(word in prediction.split() for word in negation_words)
    
    return (ground_truth in prediction) and (not contains_negation if is_cf else True)

def get_score(preds, golds, origs):
    em, gold_recall, orig_recall = 0, 0, 0
    for pred, gold, orig in zip(preds, golds, origs):
        if isinstance(gold, list):
            _em, _recall = 0, 0
            for g in gold:
                _em = max(exact_match_score(pred, g, True), _em)
                _recall = max(recall_score(pred, g, True), _recall)
        else:
            _em = exact_match_score(pred, gold, True)
            _recall = recall_score(pred, gold, True)
        if isinstance(orig, list):
            _recall_orig = 0
            for o in orig:
                _recall_orig = max(recall_score(pred, o, False), _recall_orig)
        else:
            _recall_orig = recall_score(pred, orig, False)
        em += _em
        gold_recall += _recall and not _recall_orig
        orig_recall +=  _recall_orig
        
    em = em * 100 / (len(preds) + 1e-5)
    gold_recall = gold_recall * 100 / (len(preds) + 1e-5)
    orig_recall = orig_recall * 100 / (len(preds) + 1e-5)
    return em, gold_recall, orig_recall

def qa_to_prompt(query, context):
    prompt = '{}\nQ: {}\nA: '.format(context, query)
    return prompt

def qa_to_prompt_baseline(query, context, schema):
    def get_prompt(query, context, schema, answer=''):
        if schema == 'base':
            prompt = '{}\nQ:{}\nA:{}'.format(context, query, answer)
        elif schema == 'opin':
            context = context.replace('"', "")
            prompt = 'Bob said "{}"\nQ: {} in Bob\'s opinion?\nA:{}'.format(context, query[:-1], answer)
        elif schema == 'instr+opin':
            context = context.replace('"', "")
            prompt = 'Bob said "{}"\nQ: {} in Bob\'s opinion?\nA:{}'.format(context, query[:-1], answer)
        elif schema == 'attr':
            prompt = '{}\nQ:{} based on the given tex?\nA:{}'.format(context, query[:-1], answer)
        elif schema == 'instr':
            prompt = '{}\nQ:{}\nA:{}'.format(context, query, answer)
        return prompt
    prompt = ''
    if schema in ('instr', 'instr+opin'):
        prompt = 'Instruction: read the given information and answer the corresponding question.\n\n'
    prompt = prompt + get_prompt(query, context, schema=schema)
    return prompt

    
def eval(pred_answers, orig_answers, gold_answers, step):
    em, ps, po = get_score(pred_answers, gold_answers, orig_answers)
    mr = po / (ps + po + 1e-10) * 100
    logging.info('Step: {}: ps {}, po {}, mr {}, em {}.'.format(step, ps, po, mr, em))
    
def create_log_path(log_path):
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('') 
        logging.info(f"Log file {log_path} created.")
    else:
        logging.info(f"Log file {log_path} already exists.")

def main():
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="./Models/Qwen2-7B-Instruct", type=str)
    args = parser.parse_args()
    model_name = args.model_name
    parser.add_argument("--data_path", default="./ConFiQA/ConFiQA-QA.json", type=str)
    parser.add_argument("--schema", default="base", type=str, help="Choose from the following prompting templates: base, attr, instr, opin, instr+opin.")
    args = parser.parse_args()
    schema = args.schema
    parser.add_argument("--output_path", default='./result/Qwen2-7B-Instruct.json', type=str)
    parser.add_argument("--log_path", default='./log_ConFiQA/Qwen2-7B-Instruct.log' % schema, type=str)
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_path),  # 写入日志文件
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    logging.info("Evaluate Context-Faithfulness for the Model: %s" % model_name)
    with open(args.data_path, 'r') as fh:
        data = json.load(fh)
    logging.info('Loaded {} instances.'.format(len(data)))
    
    create_log_path(args.log_path)
    
    gold_answers, pred_answers, orig_answers = [], [], []
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", low_cpu_mem_usage = True, torch_dtype=torch.float16, trust_remote_code=True)
    model.cuda()
    stop = []
    stopping_criteria = set_stop_words(tokenizer, stop)
    step = 0
    
    for d in tqdm(data):
        step += 1
        query = d['question']
        context = d['cf_context']
        cf_answer = d['cf_answer']
        orig_answer = d['orig_answer']
        
        # prompt = qa_to_prompt(query, context)
        prompt = qa_to_prompt_baseline(query, context, schema=args.schema)
        pred = call_llama(model, tokenizer, prompt, stopping_criteria, stop)
        pred_answers.append(pred)
        if len(d['cf_alias']) != 0:
            cf_answer = [cf_answer] + d['cf_alias']
        if len(d['orig_alias']) != 0:
            orig_answer = [orig_answer] + d['orig_alias']
        gold_answers.append(cf_answer)
        orig_answers.append(orig_answer)
        d['pred'] = pred
        
        if step % 100 == 0:
            eval(pred_answers, orig_answers, gold_answers, step)
        
    with open(args.output_path, 'w') as fh:
        json.dump(data, fh)
    

if __name__ == '__main__':
    main()
