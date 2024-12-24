import json
import re

import requests
from tqdm import tqdm


# 从relations.json读取允许的关系
def load_relations(relations_file):
    with open(relations_file, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def is_valid_triple(value):
    # 过滤掉以 "http" 开头的值
    if value.startswith("http"):
        return False
    # 过滤掉匹配 "Q{数字}" 的值
    if re.match(r"^Q\d+$", value):
        return False
    return True

# SPARQL查询Wikidata triples，考虑指定的关系
def fetch_wikidata_triples(wikidata_id, allowed_properties):
    url = 'https://query.wikidata.org/sparql'
    # 将允许的关系转为 SPARQL 查询的格式
    property_filter = ' '.join([f'wdt:{prop}' for prop in allowed_properties])

    query = f"""
    SELECT ?property ?propertyLabel ?value ?valueLabel WHERE {{
        wd:{wikidata_id} ?p ?value .
        VALUES ?p {{ {property_filter} }}  # 仅获取指定的关系
        ?property wikibase:directClaim ?p .
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    """
    
    headers = {
        'User-Agent': 'DataAnalysisScript/1.0 (http://mywebsite.com; myemail@example.com)',
        'Accept': 'application/sparql-results+json'
    }
    
    try:
        response = requests.get(url, headers=headers, params={'query': query, 'format': 'json'})
        response.raise_for_status()
        data = response.json()

        triples = []
        for result in data['results']['bindings']:
            property_id = result['property']['value']  # 获取关系 ID
            property_label = result['propertyLabel']['value']
            value_id = result['value']['value']  # 获取值的实体 ID
            value_label = result['valueLabel']['value']
            
            if not is_valid_triple(value_label):
                continue

            # 提取 QID
            property_id = property_id.split('/')[-1]  # 例如 'P57'
            value_id = value_id.split('/')[-1]  # 例如 'Q1611'

            triples.append({
                'property_id': property_id,
                'property_label': property_label,
                'value_id': value_id,
                'value_label': value_label
            })
        return triples
    except requests.RequestException as e:
        print(f"Error fetching triples for {wikidata_id}: {e}")
        return None

# 遍历 JSON 文件，查询并保存 triples
def save_wikidata_triples(input_json, relations_file, output_file, error_log_file):
    triples_data = {}
    errors = []

    # 读取关系文件
    allowed_properties = load_relations(relations_file).keys()

    # 读取输入的 JSON 文件
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 遍历每个实体，查询 triples

    with tqdm(total=len(data), desc="Fetching triples") as pbar:
        for entity, details in data.items():
            wikidata_id = details['wikidata_id']
            try:
                triples = fetch_wikidata_triples(wikidata_id, allowed_properties)
                if triples:
                    triples_data[entity] = {
                        'wikidata_id': wikidata_id,
                        'triples': triples
                    }
                else:
                    errors.append(f"Failed to fetch triples for {entity} ({wikidata_id})")
            except Exception as e:
                errors.append(f"Error processing {entity} ({wikidata_id}): {str(e)}")

            pbar.update(1)


    # 保存 triples 到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(triples_data, f, indent=4, ensure_ascii=False)
    print(f"Saved triples for {len(triples_data)} entities to {output_file}")

    # 保存错误日志
    with open(error_log_file, 'w', encoding='utf-8') as f:
        for error in errors:
            f.write(error + '\n')
    print(f"Logged {len(errors)} errors to {error_log_file}")

# 执行程序
if __name__ == "__main__":
    input_json = 'data_process/wikidata_entities_with_links.json'  # 输入的 JSON 文件
    relations_file = 'data_process/relations.json'  # 关系文件
    output_file = 'data_process/triples_ini.json'  # 输出的 triples 文件
    error_log_file = 'data_process/triples_error_log.txt'  # 错误日志文件
    save_wikidata_triples(input_json, relations_file, output_file, error_log_file)
