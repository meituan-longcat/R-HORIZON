import argparse 
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import threading
import os
import requests
import time
from utils import request_response



def inference(queries, fo, config, max_workers = 1):
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for key, value in tqdm(queries):
            futures.append((executor.submit(request_response, key, value, config)))

        for future in tqdm(concurrent.futures.as_completed(futures)):
            result = None
            try:
                result = future.result()
                item = {
                    'key' : result['key'],
                    'response' : result['response']
                }
                with lock:
                    fo.write(json.dumps(item, ensure_ascii = False) + '\n')
                    fo.flush()
                # print(f"完成处理: {result}")
            except Exception as e:
                print(f"task fail: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default='output.json')
    parser.add_argument('--config', type=str, default='evaluation/config.json')
    parser.add_argument('--model_name', type=str, default='gpt-4.1')
    args = parser.parse_args()
    print(args)
    model_configs = json.load(open(args.config, 'r'))
    assert "extract" in model_configs 
    assert args.model_name in model_configs['extract']
    config = model_configs['extract'][args.model_name]

    exists = set()
    if os.path.exists(args.output):
        for line in tqdm(open(args.output)):
            item = json.loads(line)
            key = item['key']
            exists.add(key)
    print(f"load exists {len(exists)} from {args.output}")

    items = [] 
    with open(args.input, 'r') as f:
        for line in f:
            items.append(json.loads(line))
    query_lst = []
    cnt = 0

    system_prompt = "You are a helpful assistant. 提取给出的结果中，每个问题的题号和答案，并返回一个json格式。key为题目序号，value为答案。"
    for item in items:
        key = item['key']
        if key in exists:
            continue
        prompt = item['response']
        prompt = prompt.split('</think>')[-1]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        query_lst.append((key, messages))
        cnt += 1
    print(f"{cnt} in total, load {len(query_lst)} new queries")

    with open(args.output, "a") as fo:
        inference(query_lst, fo, config)
