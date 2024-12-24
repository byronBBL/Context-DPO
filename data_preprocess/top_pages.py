import os
from collections import defaultdict

import requests
from tqdm import tqdm


# 获取访问量前1000的维基百科页面
def fetch_top_1000_pages(year, month):
    url = f'https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/{year}/{month:02d}/all-days'
    headers = {
        'User-Agent': 'DataAnalysisScript/1.0 (http://mywebsite.com; myemail@example.com)'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        articles = data['items'][0]['articles']
        return [article['article'] for article in articles]
    else:
        print(f"Failed to fetch data for {year}-{month}. Status code: {response.status_code}")
        return []

# 获取页面的超链接数量
def count_links_in_page(page_title):
    url = f'https://en.wikipedia.org/w/api.php'
    params = {
        'action': 'parse',
        'page': page_title,
        'prop': 'links',
        'format': 'json',
        'pllimit': 'max'  # 获取最多的链接
    }
    headers = {
        'User-Agent': 'DataAnalysisScript/1.0 (http://mywebsite.com; myemail@example.com)'
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'parse' in data:
            return len(data['parse'].get('links', []))
    return 0  # 如果请求失败或没有找到链接，则返回0

# 保存去重后的实体列表及其超链接数
def save_unique_entities_with_link_counts(years, output_file):
    all_top_pages = defaultdict(int)

    # 收集并去重
    print("Collecting pages...")
    for year in tqdm(years):
        for month in range(12, 13):
            top_pages = fetch_top_1000_pages(year, month)
            for page in top_pages:
                all_top_pages[page] += 1
    
    unique_pages = set(all_top_pages.keys())

    # 统计每个页面的超链接数并处理异常
    print("Counting links...")
    page_link_counts = {}
    for page in tqdm(unique_pages):
        try:
            link_count = count_links_in_page(page)
            page_link_counts[page] = link_count
            print(f"{page}: {link_count} links")  # 输出每个页面的链接数量
            
            # 每处理完一个页面就保存一次
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"{page}: {link_count}\n")
        
        except Exception as e:
            print(f"Error processing page '{page}': {e}")
            continue  # 继续处理下一个页面

    # 按照超链接数排序
    sorted_pages = sorted(page_link_counts.items(), key=lambda item: item[1], reverse=True)

    final_output = 'entities_with_link_counts_sorted.txt'
    
    # 将最终的结果保存到文件
    if not os.path.exists(final_output):
        with open(final_output, 'w', encoding='utf-8') as f:
            for page, link_count in sorted_pages:
                f.write(f"{page}: {link_count}\n")
    
    print(f"Saved {len(sorted_pages)} unique entities with link counts to {final_output}")

# 执行程序
if __name__ == "__main__":
    years = list(range(2023, 2024))
    output_file = 'entities_with_link_counts.txt'
    save_unique_entities_with_link_counts(years, output_file)
