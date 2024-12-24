import json
import random


# 读取JSON数据
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 过滤并处理 triples 数据
def filter_and_process_triples(fact_data, rele_template):
    filtered_data = {}
    cnt = 0
    for entity, details in fact_data.items():
        # 筛选出符合条件的 triples（根据 rele_template）
        triples = [triple for triple in details["triples"] if triple["property_id"] in rele_template]
        
        # 创建一个字典，以 property_label 为键，确保每种 property_label 仅保留一个 triple
        unique_triples = {}
        for triple in triples:
            property_label = triple["property_label"]
            if property_label not in unique_triples:
                unique_triples[property_label] = triple
            else:
                # 随机选择保留一个 triple（如果已存在，50%概率替换）
                if random.choice([True, False]):
                    unique_triples[property_label] = triple

        # 将 unique_triples 转换为列表，随机选择最多 10 个 triple
        selected_triples = list(unique_triples.values())
        if len(selected_triples) > 10:
            selected_triples = random.sample(selected_triples, 10)
        
        # 如果过滤后 triple 数量大于等于 5，则保留该实体
        if len(selected_triples) >= 5:
            filtered_data[entity] = {
                "wikidata_id": details["wikidata_id"],
                "triples": selected_triples
            }
            cnt += len(selected_triples)
    
    return filtered_data

# 示例用法
fact_data = load_json("data_process/triples_ini.json")
rele_template = load_json("rela_template.json")  # 替换为实际的属性模板

# 过滤数据
filtered_fact_data = filter_and_process_triples(fact_data, rele_template)

# 输出过滤后的数据到新文件
with open("data_process/triples_filtered.json", 'w', encoding='utf-8') as f:
    json.dump(filtered_fact_data, f, ensure_ascii=False, indent=4)