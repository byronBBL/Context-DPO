import json
from collections import defaultdict, deque

from tqdm import tqdm


# 从 JSON 文件加载数据
def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

# 构建知识图谱
def build_knowledge_graph(data):
    graph = defaultdict(list)
    map_to_label = {}
    cnt = 0
    for entity, entity_data in data.items():
        entity_name = entity.replace('_', ' ')
        for triple in entity_data['triples']:
            # 使用三元组形式 (主体, 关系, 目标)
            graph[entity_data['wikidata_id']].append((entity_data['wikidata_id'], triple['property_id'], triple['value_id']))
            
            map_to_label[triple['property_id']] = triple.get('property_label', 'Unknown')  # 默认标签为 'Unknown'
            map_to_label[triple['value_id']] = triple.get('value_label', 'Unknown')  # 默认标签为 'Unknown'
            map_to_label[entity_data['wikidata_id']] = entity_name  # 默认标签为 'Unknown'
            cnt += 1
    return graph, map_to_label

def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def find_k_hop_paths(graph, map_to_label, max_hops):
    all_paths = defaultdict(lambda: defaultdict(list))
    all_paths_labeled = defaultdict(lambda: defaultdict(list))
    rela_paths_entities = {}
    rela_paths = []
    rela_paths.append(defaultdict(list))  # rela_paths[0]
    rela_paths.append(defaultdict(lambda: defaultdict(list)))  # rela_paths[1]
    rela_paths.append(defaultdict(lambda: defaultdict(lambda: defaultdict(list))))  # rela_paths[2]
    rela_paths.append(defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))  # rela_paths[3]
    
    # 获取图中所有实体的列表
    entities = list(graph.keys())
    
    # 遍历每个实体
    for entity in tqdm(entities):
        
        for k in range(0, max_hops + 1):
            queue = deque([(entity, [], 0)])  # (当前实体, 当前路径, 当前跳数)

            while queue:
                current_entity, current_path, current_hop = queue.popleft()

                if current_hop < k:  # 仅在当前跳数小于 k 时继续
                    for subject, property_id, value_id in graph[current_entity]:
                        # 检查路径中是否已有实体和关系
                        if value_id not in [value for _, _, value in current_path] and value_id not in [value for value, _, _  in current_path] and property_id not in [prop for _, prop, _ in current_path]:
                            if current_path != [] and value_id == current_path[0][0]: #cycle
                                continue
                                
                            new_path = current_path + [(subject, property_id, value_id)]  # 添加三元组到路径
                            queue.append((value_id, new_path, current_hop + 1))

                            # 存储路径
                            if current_hop + 1 == k:
                                new_path_str = repr(new_path)
                                all_paths[entity][current_hop].append(new_path_str)
                                labeled_path = [(map_to_label.get(subject, subject), map_to_label.get(property_id, property_id), map_to_label.get(value_id, value_id)) for subject, property_id, value_id in new_path]
                                all_paths_labeled[entity][current_hop].append(repr(labeled_path))
                                if current_hop == 0:
                                    rela_paths[current_hop][new_path[0][1]].append(new_path_str)
                                elif current_hop == 1:
                                    rela_paths[current_hop][new_path[0][1]][new_path[1][1]].append(new_path_str)
                                elif current_hop == 2:
                                    rela_paths[current_hop][new_path[0][1]][new_path[1][1]][new_path[2][1]].append(new_path_str)
                                elif current_hop == 3:
                                    rela_paths[current_hop][new_path[0][1]][new_path[1][1]][new_path[2][1]][new_path[3][1]].append(new_path_str)
                                
                            if current_hop == 2:
                                a = 1

    return all_paths, all_paths_labeled, rela_paths

# 主程序
if __name__ == "__main__":
    data = load_data('data_process/triples_filtered.json')
    knowledge_graph, map_to_label = build_knowledge_graph(data)
    max_k = 4 
    k_hop_paths, k_hop_paths_labeled, rela_paths_entities = find_k_hop_paths(knowledge_graph, map_to_label, max_k)
    
    save_to_json(k_hop_paths, 'data_process/entities_paths.json')
    save_to_json(map_to_label, 'data_process/map_to_label.json')
    save_to_json(rela_paths_entities, 'data_process/entities_paths_rela.json')