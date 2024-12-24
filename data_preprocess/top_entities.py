import json

import requests
from tqdm import tqdm


# 查询每个维基百科页面对应的 Wikidata 实体编号
def get_wikidata_id(page_title):
    url = f'https://en.wikipedia.org/w/api.php'
    params = {
        'action': 'query',
        'format': 'json',
        'titles': page_title,
        'prop': 'pageprops',
        'ppprop': 'wikibase_item',
    }
    headers = {
        'User-Agent': 'DataAnalysisScript/1.0 (http://mywebsite.com; myemail@example.com)'
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # 检查HTTP错误
        data = response.json()
        pages = data['query']['pages']
        for page_id, page_info in pages.items():
            if 'pageprops' in page_info and 'wikibase_item' in page_info['pageprops']:
                return page_info['pageprops']['wikibase_item']
    except requests.RequestException as e:
        print(f"Error fetching Wikidata ID for page '{page_title}': {e}")
    
    return None

# 读取实体列表并匹配 Wikidata 实体编号，同时保留超链接数
def match_and_save_wikidata_ids(input_file, output_file, error_log_file):
    wiki_to_wikidata = {}
    errors = []
    total_count = 20888  # 设置总数

    # 读取实体列表并提取页面标题和超链接数
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    with open('wikidata_entities_with_links.json', 'r', encoding='utf-8') as f:
        wiki = json.load(f)

    with tqdm(total=total_count, desc="Processing Pages", unit="page") as pbar:
        for cnt, line in enumerate(lines, start=1):
            pbar.update(1)
            if ':' in line:
                page, link_count = line.rsplit(':', 1)  # 根据最后一个冒号进行拆分
                page = page.strip()
                link_count = int(link_count.strip())  # 将超链接数转换为整数
                if page in wiki:
                    wiki_to_wikidata[page] = wiki[page]
                    continue
                # 查询并匹配 Wikidata 实体编号
                try:
                    wikidata_id = get_wikidata_id(page)
                    if wikidata_id:
                        wiki_to_wikidata[page] = {
                            "wikidata_id": wikidata_id,
                            "links": link_count
                        }
                    else:
                        errors.append(f"Failed to get Wikidata ID for page: {page}, {cnt}")
                except Exception as e:
                    errors.append(f"Error processing page '{page}. {cnt}': {str(e)}")
            

    # 将匹配结果保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(wiki_to_wikidata, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(wiki_to_wikidata)} Wikidata IDs with link counts to {output_file}")

    # 将出错的页面写入错误日志文件
    with open(error_log_file, 'w', encoding='utf-8') as f:
        for error in errors:
            f.write(error + '\n')
    print(f"Logged {len(errors)} errors to {error_log_file}")

# 执行程序
if __name__ == "__main__":
    input_file = 'entities_with_link_counts_sorted.txt'  # 输入文件
    output_file = 'wikidata_entities_with_links.json'  # 输出文件
    error_log_file = 'error_log.txt'  # 错误日志文件
    match_and_save_wikidata_ids(input_file, output_file, error_log_file)
