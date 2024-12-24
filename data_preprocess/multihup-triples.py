import ast
import json
import random
from collections import defaultdict, deque
from copy import deepcopy

from tqdm import tqdm


# 从 JSON 文件加载数据
def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        
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

def convert_triples_to_labels(triples, map_to_label):
    labeled_triples = []
    
    for subject, predicate, obj in triples:
        # 获取每个 ID 对应的标签，如果没有对应标签则返回原 ID
        subject_label = map_to_label.get(subject, subject)
        predicate_label = map_to_label.get(predicate, predicate)
        obj_label = map_to_label.get(obj, obj)
        
        # 将转换后的三元组添加到新列表
        labeled_triples.append((subject_label, predicate_label, obj_label))
    
    return labeled_triples

def get_cf_paths(hop_paths, rela_paths, map_to_label):
    
    MR_lst = {}
    data_mc = {}
    
    for hop_num, hop_path_lst in hop_paths.items():
        # if hop_num != '3': continue
        MR_lst[str(hop_num)] = []
        for hop_path in tqdm(hop_path_lst):
            cf_path = deepcopy(hop_path)
            position_sample_lst = [i for i in range(len(cf_path))]
            
            while len(position_sample_lst) != 0:
                cf_position = random.choice(position_sample_lst)
                meta_path = [p[1] for p in cf_path[cf_position:]]
                rela_path_lst = rela_paths[len(meta_path) - 1]
                for rela in meta_path:
                    rela_path_lst = rela_path_lst[rela]
                rela_path_lst = [p for p in rela_path_lst if p[0][-1] != cf_path[cf_position][-1]]
                if len(rela_path_lst) == 0:
                    position_sample_lst.remove(cf_position)
                    continue
                sampled_path = random.choice(rela_path_lst)
                tmp = list(cf_path[cf_position])
                tmp[-1] = sampled_path[0][-1]
                cf_path[cf_position] = tuple(tmp)
                if len(sampled_path) > 1: 
                    cf_path[cf_position + 1:] = sampled_path[1:]
                
                hop_path_labeled = convert_triples_to_labels(hop_path, map_to_label)
                cf_path_labeled = convert_triples_to_labels(cf_path, map_to_label)
                MR_lst[str(hop_num)].append({"orig_path": repr(hop_path), "cf_path": repr(cf_path), "orig_path_labeled": repr(hop_path_labeled), "cf_path_labeled": repr(cf_path_labeled), "orig_triple": repr(hop_path[cf_position]), "cf_triple": repr(cf_path[cf_position])})
                break
            
        if hop_num == '0': continue
        data_mc[str(int(hop_num) + 1)+'-hop'] = []
        for hop_path in tqdm(hop_path_lst):
            orig = [hop_path[0]]
            cf_path = deepcopy(hop_path)
            meta_path = [p[1] for p in hop_path]
            
            for pos in range(len(cf_path) - 1):
                rela1 = meta_path[pos]
                rela2 = meta_path[pos + 1]
                rela_path_lst = rela_paths[1][rela1][rela2]
                rela_path_lst = [p for p in rela_path_lst if p[0][-1] != cf_path[pos][-1]]
                if len(rela_path_lst) == 0:
                    continue
                sampled_path = random.choice(rela_path_lst)
                tmp1 = list(cf_path[pos])
                tmp2 = list(cf_path[pos+1])
                tmp1[-1] = sampled_path[0][-1]
                tmp2[-1] = sampled_path[1][-1]
                tmp2[0] = sampled_path[1][0]
                cf_path[pos] = tuple(tmp1)
                cf_path[pos+1] = tuple(tmp2)
                orig.append(sampled_path[-1])
            rela0 = meta_path[-1]
            rela_path_lst = rela_paths[0][rela0]
            rela_path_lst = [p for p in rela_path_lst if p[0][-1] != cf_path[-1][-1]]
            if len(rela_path_lst) == 0:
                continue
            sampled_path = random.choice(rela_path_lst)
            tmp = list(cf_path[-1])
            tmp[-1] = sampled_path[0][-1]
            cf_path[-1] = tuple(tmp)
            
            hop_path_labeled = convert_triples_to_labels(hop_path, map_to_label)
            cf_path_labeled = convert_triples_to_labels(cf_path, map_to_label)
            data_mc[str(int(hop_num) + 1)+'-hop'].append({"orig_path": repr(hop_path), "cf_path": repr(cf_path), "orig_path_labeled": repr(hop_path_labeled), "cf_path_labeled": repr(cf_path_labeled), "orig": repr(orig)})
            
            
        
    return MR_lst, data_mc
        
if __name__ == "__main__":
    # 加载数据
    rela_paths = load_data('data_process/entities_paths_rela.json')
    entity_paths = load_data('data_process/entities_paths.json')
    map_to_label = load_data('data_process/map_to_label.json')
    
    map_to_label['P413'] = 'position played'
    map_to_label['P140'] = 'religion'
    
    rela_paths = convert_str_to_list(rela_paths)
    entity_paths = convert_str_to_list(entity_paths)
    
    hop_paths = {"0":[], "1":[], "2":[], "3":[]}
    data_mr = {"2-hop":[], "3-hop":[], "4-hop":[]}
    
    for entity, hop_data in entity_paths.items():
        for hop_num, paths in hop_data.items():
            hop_paths[hop_num] += paths
            a = 1
            
    MR_lst, data_mc = get_cf_paths(hop_paths, rela_paths, map_to_label)
    data_qa = random.sample(MR_lst['0'], 12000)
    
    for hop_num, paths in MR_lst.items(): 
        if hop_num == '0': continue
        data_mr[str(int(hop_num) + 1)+'-hop'] = random.sample(paths, 4000) #每个hop采4000paths
        
    for hop_num, paths in data_mc.items(): 
        data_mc[hop_num] = random.sample(paths, 4000) #每个hop采4000paths
    
    save_to_json(data_qa, 'data_process/multihopdata_qa_triple.json')
    save_to_json(data_mr, 'data_process/multihopdata_mr_triple.json')
    save_to_json(data_mc, 'data_process/multihopdata_mc_triple.json')
    
    
    